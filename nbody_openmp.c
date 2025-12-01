
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <omp.h>


#define INPUT_FILE "taxi_nbody_input.csv"
#define OUTPUT_FILE "nbody_results_openmp.csv"


#define WINDOW_SIZE 10000


#define ALPHA 1.0
#define BETA 0.5
#define R 6371.0

typedef struct {
    char id[32];
    double lat;
    double lon;
    long time_min;
    double congestion_score;
} Particle;


double to_radians(double deg) {
    return deg * (M_PI / 180.0);
}

double haversine_km(double lat1, double lon1, double lat2, double lon2) {
    double dLat = to_radians(lat2 - lat1);
    double dLon = to_radians(lon2 - lon1);
    double a = sin(dLat / 2) * sin(dLat / 2) +
               cos(to_radians(lat1)) * cos(to_radians(lat2)) *
               sin(dLon / 2) * sin(dLon / 2);
    double c = 2 * atan2(sqrt(a), sqrt(1 - a));
    return R * c;
}


int main() {
    printf("=== NYC Taxi N-Body (OpenMP) ===\n");
    printf("Window Size: +/- %d neighbors\n", WINDOW_SIZE);

    FILE *fp = fopen(INPUT_FILE, "r");
    if (!fp) {
        printf("Error: Run python script first or place %s in cwd.\n", INPUT_FILE);
        return 1;
    }


    long n_particles = 0;
    char buffer[1024];
    while (fgets(buffer, sizeof(buffer), fp)) n_particles++;
    rewind(fp);

    printf("Loading %ld particles...\n", n_particles);

    Particle *particles = (Particle *)malloc(n_particles * sizeof(Particle));
    if (!particles) { printf("Memory allocation failed.\n"); fclose(fp); return 1; }


    long idx = 0;
    while (fgets(buffer, sizeof(buffer), fp)) {
        if (sscanf(buffer, "%31[^,],%lf,%lf,%ld",
                   particles[idx].id,
                   &particles[idx].lat,
                   &particles[idx].lon,
                   &particles[idx].time_min) == 4) {
            particles[idx].congestion_score = 0.0;
            idx++;
        }
    }
    fclose(fp);

    printf("Starting parallel simulation with %d threads...\n", omp_get_max_threads());
    double t_start = omp_get_wtime();


    long counter = 0;

    #pragma omp parallel for schedule(dynamic, 64)
    for (long i = 0; i < n_particles; i++) {
        long start_j = (i - WINDOW_SIZE < 0) ? 0 : i - WINDOW_SIZE;
        long end_j   = (i + WINDOW_SIZE >= n_particles) ? n_particles : i + WINDOW_SIZE + 1;

        double local_score = 0.0;

        for (long j = start_j; j < end_j; j++) {
            if (i == j) continue;
            double t_diff = (double)llabs(particles[i].time_min - particles[j].time_min);
            double d_dist = haversine_km(particles[i].lat, particles[i].lon,
                                         particles[j].lat, particles[j].lon);
            double denominator = 1.0 + (ALPHA * d_dist) + (BETA * t_diff);
            local_score += 1.0 / denominator;
        }

        particles[i].congestion_score = local_score;


        #pragma omp atomic
        counter++;

        if (counter % (n_particles / 20 + 1) == 0) { 
            #pragma omp critical
            {
                double pct = (double)counter / (double)n_particles * 100.0;
                printf("\rProgress: %.1f%%", pct);
                fflush(stdout);
            }
        }
    } 

    double t_end = omp_get_wtime();
    printf("\rProgress: 100.0%%\n");
    printf("\n------------------------------------------------\n");
    printf("OpenMP Processing Time: %.4f seconds\n", t_end - t_start);
    printf("Total Interactions Processed: ~%.2e\n", (double)n_particles * WINDOW_SIZE * 2);
    printf("------------------------------------------------\n");

   
    printf("Saving results to %s...\n", OUTPUT_FILE);
    FILE *out = fopen(OUTPUT_FILE, "w");
    if (!out) { perror("fopen"); free(particles); return 1; }
    fprintf(out, "id,lat,lon,time_min,congestion_score\n");
    for (long i = 0; i < n_particles; i++) {
        fprintf(out, "%s,%.5f,%.5f,%ld,%.6f\n",
                particles[i].id, particles[i].lat, particles[i].lon,
                particles[i].time_min, particles[i].congestion_score);
    }
    fclose(out);
    free(particles);

    return 0;
}
