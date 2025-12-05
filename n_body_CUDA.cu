#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <time.h>

// --- Configuration ---
#define INPUT_FILE "taxi_nbody_input.csv"
#define OUTPUT_FILE "nbody_cuda_results.csv"

// We use a smaller window for testing, or keep 10000 for the benchmark
#define WINDOW_SIZE 10000 
#define BLOCK_SIZE 256

// --- Constant Memory ---
// Storing these in constant memory allows broadcast reading (very fast)
__constant__ float d_ALPHA;
__constant__ float d_BETA;
__constant__ float d_R;
__constant__ float d_PI_DIV_180;

// --- Data Structure ---
// Using float instead of double for GPU performance
typedef struct {
    char id[32]; // Keeping ID for printing, though not used in calc
    float lat;
    float lon;
    int time_min; 
    float congestion_score;
} Particle;

// --- Helper: CUDA Error Checking ---
// Standard macro to catch CUDA errors during development
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// --- The Kernel ---
__global__ void compute_congestion_kernel(Particle* particles, int n) {
    
    // Calculate global thread ID
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) return;

    // Load 'my' particle data into registers
    // This avoids reading global memory repeatedly for the current particle
    float my_lat = particles[i].lat;
    float my_lon = particles[i].lon;
    int my_time = particles[i].time_min;
    
    // Pre-convert my coordinates to radians
    float lat1_rad = my_lat * d_PI_DIV_180;
    float lon1_rad = my_lon * d_PI_DIV_180;
    float cos_lat1 = __cosf(lat1_rad); // Intrinsic function

    // Define Window Boundaries
    int start_j = (i - WINDOW_SIZE < 0) ? 0 : i - WINDOW_SIZE;
    int end_j = (i + WINDOW_SIZE >= n) ? n : i + WINDOW_SIZE + 1;

    float local_score = 0.0f;

    // The Sliding Window Loop
    for (int j = start_j; j < end_j; j++) {
        if (i == j) continue;

        // Use __ldg() to force Read-Only Cache.
        // This is crucial because neighbors (threads) read neighbors (data).
        float other_lat = __ldg(&particles[j].lat);
        float other_lon = __ldg(&particles[j].lon);
        int other_time = __ldg(&particles[j].time_min);

        // 1. Time Diff (Absolute)
        float t_diff = (float)abs(my_time - other_time);

        // 2. Spatial Distance (Haversine Optimized)
        float lat2_rad = other_lat * d_PI_DIV_180;
        float lon2_rad = other_lon * d_PI_DIV_180;

        float dlat = lat2_rad - lat1_rad;
        float dlon = lon2_rad - lon1_rad;

        // Use fast math intrinsics (__sinf, __sqrtf)
        float sin_dlat = __sinf(dlat * 0.5f);
        float sin_dlon = __sinf(dlon * 0.5f);

        float a = sin_dlat * sin_dlat + 
                  cos_lat1 * __cosf(lat2_rad) * sin_dlon * sin_dlon;
        
        // Clamping to prevent numerical errors with sqrt
        if (a < 0.0f) a = 0.0f;
        if (a > 1.0f) a = 1.0f;

        // Optimized atan2 logic for distance
        float c = 2.0f * atan2f(__fsqrt_rn(a), __fsqrt_rn(1.0f - a));
        float d_dist = d_R * c;

        // 3. Kernel Calculation
        float denom = 1.0f + (d_ALPHA * d_dist) + (d_BETA * t_diff);
        local_score += (1.0f / denom);
    }

    // Write back result
    particles[i].congestion_score = local_score;
}

// --- Host Main ---
int main() {
    printf("=== CUDA N-Body Simulation ===\n");

    // 1. Set Constants on Host
    float h_alpha = 1.0f;
    float h_beta = 0.5f;
    float h_R = 6371.0f;
    float h_pi_div = 3.14159265359f / 180.0f;

    // Copy to GPU Constant Memory
    gpuErrchk(cudaMemcpyToSymbol(d_ALPHA, &h_alpha, sizeof(float)));
    gpuErrchk(cudaMemcpyToSymbol(d_BETA, &h_beta, sizeof(float)));
    gpuErrchk(cudaMemcpyToSymbol(d_R, &h_R, sizeof(float)));
    gpuErrchk(cudaMemcpyToSymbol(d_PI_DIV_180, &h_pi_div, sizeof(float)));

    // 2. Load Data (Sequential Host Code)
    FILE *fp = fopen(INPUT_FILE, "r");
    if (!fp) { printf("Error opening input file.\n"); return 1; }

    // Count lines
    int n_particles = 0;
    char buffer[1024];
    while (fgets(buffer, 1024, fp)) n_particles++;
    rewind(fp);
    printf("Particles: %d\n", n_particles);

    // Allocate Host Memory
    size_t bytes = n_particles * sizeof(Particle);
    Particle *h_particles = (Particle*)malloc(bytes);
    
    // Read Data
    int idx = 0;
    while (fgets(buffer, 1024, fp)) {
        // Parsing logic matching the Python script output
        if (sscanf(buffer, "%[^,],%f,%f,%d", 
            h_particles[idx].id, 
            &h_particles[idx].lat, 
            &h_particles[idx].lon, 
            &h_particles[idx].time_min) == 4) {
            h_particles[idx].congestion_score = 0.0f;
            idx++;
        }
    }
    fclose(fp);

    // 3. GPU Memory Setup
    Particle *d_particles;
    gpuErrchk(cudaMalloc(&d_particles, bytes));
    
    // Copy Data Host -> Device
    printf("Copying data to GPU...\n");
    gpuErrchk(cudaMemcpy(d_particles, h_particles, bytes, cudaMemcpyHostToDevice));

    // 4. Launch Configuration
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (n_particles + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("Launching Kernel: %d blocks, %d threads.\n", blocksPerGrid, threadsPerBlock);
    
    // Timing Start
    clock_t start = clock();
    
    compute_congestion_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_particles, n_particles);
    
    // Wait for GPU to finish
    gpuErrchk(cudaDeviceSynchronize());
    
    // Timing End
    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    // 5. Retrieve Results
    printf("Copying results back to Host...\n");
    gpuErrchk(cudaMemcpy(h_particles, d_particles, bytes, cudaMemcpyDeviceToHost));

    printf("Kernel Execution Time: %.4f seconds\n", time_taken);

    // 6. Save Results
    FILE *out = fopen(OUTPUT_FILE, "w");
    fprintf(out, "id,lat,lon,time_min,congestion_score\n");
    for (int i = 0; i < n_particles; i++) {
        fprintf(out, "%s,%.5f,%.5f,%d,%.4f\n", 
            h_particles[i].id, 
            h_particles[i].lat, 
            h_particles[i].lon, 
            h_particles[i].time_min, 
            h_particles[i].congestion_score);
    }
    fclose(out);

    // Cleanup
    cudaFree(d_particles);
    free(h_particles);
    printf("Done.\n");

    return 0;
}
