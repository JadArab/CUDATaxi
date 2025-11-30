#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

// --- Configuration ---
#define INPUT_FILE "taxi_nbody_input.csv"
#define OUTPUT_FILE "nbody_results.csv"

// --- Tuning Parameters ---
// WINDOW_SIZE: +/- neighbors to check. 
// 10,000 is chosen to create a noticeable CPU load (~1 minute runtime)
// while maintaining high accuracy for rush-hour windows.
#define WINDOW_SIZE 10000 

// --- Physics Constants ---
#define ALPHA 1.0   // Spatial Weight
#define BETA 0.5    // Temporal Weight
#define R 6371.0    // Earth Radius (km)

// --- Structures ---
typedef struct {
    char id[32];
    double lat;
    double lon;
    long time_min;
    double congestion_score;
} Particle;

// --- Helper Functions ---

double to_radians(double deg) {
    return deg * (M_PI / 180.0);
}

// Haversine Distance (The expensive part of the kernel)
double haversine_km(double lat1, double lon1, double lat2, double lon2) {
    double dLat = to_radians(lat2 - lat1);
    double dLon = to_radians(lon2 - lon1);
    double a = sin(dLat / 2) * sin(dLat / 2) +
               cos(to_radians(lat1)) * cos(to_radians(lat1)) *
               sin(dLon / 2) * sin(dLon / 2);
    double c = 2 * atan2(sqrt(a), sqrt(1 - a));
    return R * c;
}

int main() {
    printf("=== NYC Taxi N-Body Gravity Simulation (Sequential) ===\n");
    printf("Window Size: +/- %d neighbors\n", WINDOW_SIZE);
    
    // 1. Memory Allocation & Loading
    // We count lines first to allocate exact memory
    FILE *fp = fopen(INPUT_FILE, "r");
    if (!fp) { printf("Error: Run python script first.\n"); return 1; }
    
    long n_particles = 0;
    char buffer[1024];
    while (fgets(buffer, 1024, fp)) n_particles++;
    rewind(fp);

    printf("Loading %ld particles...\n", n_particles);
    
    Particle *particles = (Particle *)malloc(n_particles * sizeof(Particle));
    if (!particles) { printf("Memory allocation failed.\n"); return 1; }

    long idx = 0;
    while (fgets(buffer, 1024, fp)) {
        if (sscanf(buffer, "%[^,],%lf,%lf,%ld", 
            particles[idx].id, &particles[idx].lat, &particles[idx].lon, &particles[idx].time_min) == 4) {
            particles[idx].congestion_score = 0.0;
            idx++;
        }
    }
    fclose(fp);

    // 2. The Simulation Loop (Timed)
    printf("Starting Simulation... (This may take 1-2 minutes)\n");
    
    clock_t start = clock(); // Start Timer

    for (long i = 0; i < n_particles; i++) {
        
        // Define sliding window boundaries
        long start_j = (i - WINDOW_SIZE < 0) ? 0 : i - WINDOW_SIZE;
        long end_j = (i + WINDOW_SIZE >= n_particles) ? n_particles : i + WINDOW_SIZE + 1;

        double local_score = 0.0;

        for (long j = start_j; j < end_j; j++) {
            if (i == j) continue;

            // 1. Calculate Time Distance T (Absolute Difference)
            double t_diff = (double)labs(particles[i].time_min - particles[j].time_min);
            
            // 2. Calculate Spatial Distance D (Haversine)
            double d_dist = haversine_km(particles[i].lat, particles[i].lon, 
                                         particles[j].lat, particles[j].lon);

            // 3. The Kernel: S = 1 / (1 + aD + bT)
            double denominator = 1.0 + (ALPHA * d_dist) + (BETA * t_diff);
            local_score += (1.0 / denominator);
        }
        
        particles[i].congestion_score = local_score;

        // Progress update every 5%
        if (i % (n_particles / 20) == 0) {
            printf("\rProgress: %.1f%%", (double)i / n_particles * 100.0);
            fflush(stdout);
        }
    }

    clock_t end = clock(); // End Timer
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("\rProgress: 100.0%%\n");
    printf("\n------------------------------------------------\n");
    printf("Sequential Processing Time: %.4f seconds\n", time_taken);
    printf("Total Interactions Processed: ~%.2e\n", (double)n_particles * WINDOW_SIZE * 2);
    printf("------------------------------------------------\n");

    // 3. Export Results
    printf("Saving results to %s...\n", OUTPUT_FILE);
    FILE *out = fopen(OUTPUT_FILE, "w");
    fprintf(out, "id,lat,lon,time_min,congestion_score\n"); // Header
    for (long i = 0; i < n_particles; i++) {
        fprintf(out, "%s,%.5f,%.5f,%ld,%.4f\n", 
            particles[i].id, particles[i].lat, particles[i].lon, 
            particles[i].time_min, particles[i].congestion_score);
    }
    fclose(out);
    free(particles);

    return 0;
}
