// nbody_pthreads.c (fixed)
// - wall-clock timing (clock_gettime)
// - atomic progress counter + rare printing (mutex only for print)
// - accepts thread count via argv[1]
// - writes results to nbody_results_pthreads_<N>threads.csv

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <stdint.h>
#include <unistd.h>

#define INPUT_FILE "taxi_nbody_input.csv"
#define OUT_BASENAME "nbody_results_pthreads"
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

typedef struct {
    Particle *particles;
    long n_particles;
    long start_idx;
    long end_idx; // exclusive
    long *progress_counter;            // shared atomic counter
    long progress_report_step;         // how often to print (in increments)
    pthread_mutex_t *print_mutex;      // mutex only used for printing
} ThreadArg;

static inline double to_radians(double deg) { return deg * (M_PI / 180.0); }

double haversine_km(double lat1, double lon1, double lat2, double lon2) {
    double dLat = to_radians(lat2 - lat1);
    double dLon = to_radians(lon2 - lon1);
    double a = sin(dLat / 2) * sin(dLat / 2) +
               cos(to_radians(lat1)) * cos(to_radians(lat2)) *
               sin(dLon / 2) * sin(dLon / 2);
    double c = 2 * atan2(sqrt(a), sqrt(1 - a));
    return R * c;
}

void *worker(void *arg) {
    ThreadArg *ta = (ThreadArg *)arg;
    Particle *particles = ta->particles;

    for (long i = ta->start_idx; i < ta->end_idx; i++) {
        long start_j = (i - WINDOW_SIZE < 0) ? 0 : i - WINDOW_SIZE;
        long end_j = (i + WINDOW_SIZE >= ta->n_particles) ? ta->n_particles : i + WINDOW_SIZE + 1;

        double local_score = 0.0;
        for (long j = start_j; j < end_j; j++) {
            if (i == j) continue;
            double t_diff = (double)llabs(particles[i].time_min - particles[j].time_min);
            double d_dist = haversine_km(particles[i].lat, particles[i].lon,
                                         particles[j].lat, particles[j].lon);
            double denominator = 1.0 + (ALPHA * d_dist) + (BETA * t_diff);
            local_score += (1.0 / denominator);
        }
        particles[i].congestion_score = local_score;

        // atomic increment progress counter by 1 (lock-free)
        long prev = __atomic_add_fetch(ta->progress_counter, 1L, __ATOMIC_RELAXED);

        // rare print path: only when hitting the configured reporting step
        if (ta->progress_report_step > 0 && (prev % ta->progress_report_step == 0)) {
            pthread_mutex_lock(ta->print_mutex);
            double pct = (double)prev / (double)ta->n_particles * 100.0;
            if (pct > 100.0) pct = 100.0;
            printf("\rProgress (approx): %.1f%%", pct);
            fflush(stdout);
            pthread_mutex_unlock(ta->print_mutex);
        }
    }
    return NULL;
}

int main(int argc, char **argv) {
    printf("=== NYC Taxi N-Body (pthreads) ===\n");
    printf("Window Size: +/- %d neighbors\n", WINDOW_SIZE);

    FILE *fp = fopen(INPUT_FILE, "r");
    if (!fp) { fprintf(stderr, "Error: Cannot open %s\n", INPUT_FILE); return 1; }

    long n_particles = 0;
    char buffer[1024];
    while (fgets(buffer, sizeof(buffer), fp)) n_particles++;
    rewind(fp);

    printf("Loading %ld particles...\n", n_particles);
    Particle *particles = (Particle *)malloc(n_particles * sizeof(Particle));
    if (!particles) { fprintf(stderr, "Memory allocation failed.\n"); fclose(fp); return 1; }

    long idx = 0;
    while (fgets(buffer, sizeof(buffer), fp)) {
        if (sscanf(buffer, "%31[^,],%lf,%lf,%ld",
                   particles[idx].id, &particles[idx].lat, &particles[idx].lon, &particles[idx].time_min) == 4) {
            particles[idx].congestion_score = 0.0;
            idx++;
        }
    }
    fclose(fp);

    // determine thread count (argv[1] overrides)
    int nthreads = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (nthreads < 1) nthreads = 4;
    if (argc > 1) {
        int cmd = atoi(argv[1]);
        if (cmd > 0) nthreads = cmd;
    }
    if (nthreads < 1) nthreads = 1;

    printf("Using %d threads\n", nthreads);

    // prepare threading structures
    pthread_t *threads = malloc(sizeof(pthread_t) * nthreads);
    ThreadArg *args = malloc(sizeof(ThreadArg) * nthreads);
    if (!threads || !args) { fprintf(stderr, "Memory allocation failed for threads.\n"); free(particles); return 1; }

    pthread_mutex_t print_mutex;
    pthread_mutex_init(&print_mutex, NULL);

    long progress_counter = 0;
    // reporting step: every ~5% (but at least 1)
    long progress_step = n_particles / 20;
    if (progress_step < 1) progress_step = 1;

    long chunk = (n_particles + nthreads - 1) / nthreads;

    // start wall-clock timer
    struct timespec tstart, tend;
    clock_gettime(CLOCK_MONOTONIC, &tstart);

    // create worker threads
    for (int t = 0; t < nthreads; t++) {
        long start_idx = t * chunk;
        long end_idx = start_idx + chunk;
        if (start_idx >= n_particles) { start_idx = n_particles; end_idx = n_particles; }
        if (end_idx > n_particles) end_idx = n_particles;

        args[t].particles = particles;
        args[t].n_particles = n_particles;
        args[t].start_idx = start_idx;
        args[t].end_idx = end_idx;
        args[t].progress_counter = &progress_counter;
        args[t].progress_report_step = progress_step;
        args[t].print_mutex = &print_mutex;

        if (pthread_create(&threads[t], NULL, worker, &args[t]) != 0) {
            fprintf(stderr, "Error: pthread_create failed for thread %d\n", t);
            // join previously created threads
            for (int k = 0; k < t; k++) pthread_join(threads[k], NULL);
            pthread_mutex_destroy(&print_mutex);
            free(threads); free(args); free(particles);
            return 1;
        }
    }

    // join
    for (int t = 0; t < nthreads; t++) pthread_join(threads[t], NULL);

    // end wall-clock timer
    clock_gettime(CLOCK_MONOTONIC, &tend);
    double elapsed = (tend.tv_sec - tstart.tv_sec) + (tend.tv_nsec - tstart.tv_nsec) / 1e9;

    // final progress print
    printf("\rProgress: 100.0%%\n");
    printf("\n------------------------------------------------\n");
    printf("pthreads Processing Time: %.4f seconds\n", elapsed);
    printf("Total Interactions Processed: ~%.2e\n", (double)n_particles * WINDOW_SIZE * 2);
    printf("------------------------------------------------\n");

    // write output file named with thread count to avoid overwriting
    char out_fname[256];
    snprintf(out_fname, sizeof(out_fname), OUT_BASENAME "_%dthreads.csv", nthreads);
    printf("Saving results to %s...\n", out_fname);

    FILE *out = fopen(out_fname, "w");
    if (!out) { perror("fopen"); pthread_mutex_destroy(&print_mutex); free(threads); free(args); free(particles); return 1; }

    fprintf(out, "id,lat,lon,time_min,congestion_score\n");
    for (long i = 0; i < n_particles; i++) {
        fprintf(out, "%s,%.6f,%.6f,%ld,%.6f\n",
                particles[i].id, particles[i].lat, particles[i].lon,
                particles[i].time_min, particles[i].congestion_score);
    }
    fclose(out);

    pthread_mutex_destroy(&print_mutex);
    free(threads);
    free(args);
    free(particles);
    return 0;
}
