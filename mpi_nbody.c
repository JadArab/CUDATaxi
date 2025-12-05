// mpi_nbody.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>

#define INPUT_FILE "taxi_nbody_input.csv"
#define OUTPUT_FILE "nbody_results.csv"
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
               cos(to_radians(lat1)) * cos(to_radians(lat1)) *
               sin(dLon / 2) * sin(dLon / 2);
    double c = 2 * atan2(sqrt(a), sqrt(1 - a));
    return R * c;
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Particle *particles = NULL;
    long n_particles = 0;
    char buffer[1024];

    if (rank == 0) {
        FILE *fp = fopen(INPUT_FILE, "r");
        if (!fp) {
            fprintf(stderr, "Error: cannot open %s\n", INPUT_FILE);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        while (fgets(buffer, sizeof(buffer), fp)) n_particles++;
        rewind(fp);

        particles = (Particle*)malloc(n_particles * sizeof(Particle));
        if (!particles) {
            fprintf(stderr, "Memory allocation failed on root\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        long idx = 0;
        while (fgets(buffer, sizeof(buffer), fp)) {
            if (sscanf(buffer, "%31[^,],%lf,%lf,%ld",
                       particles[idx].id, &particles[idx].lat, &particles[idx].lon, &particles[idx].time_min) == 4) {
                particles[idx].congestion_score = 0.0;
                idx++;
            }
        }
        fclose(fp);
        // adjust n_particles in case some lines failed parse
        n_particles = idx;
        printf("Rank 0: loaded %ld particles\n", n_particles);
    }

    // Broadcast n_particles to all ranks
    MPI_Bcast(&n_particles, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    // Allocate particles on non-root ranks
    if (rank != 0) {
        particles = (Particle*)malloc(n_particles * sizeof(Particle));
        if (!particles) {
            fprintf(stderr, "Rank %d: memory allocation failed\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Broadcast raw particle bytes to all ranks (fast and simple)
    MPI_Bcast(particles, (int)(n_particles * sizeof(Particle)), MPI_BYTE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // compute distribution: block distribution with remainder
    long base = n_particles / size;
    int rem = (int)(n_particles % size);
    long start = rank * base + (rank < rem ? rank : rem);
    long count = base + (rank < rem ? 1 : 0);
    long end = start + count; // [start, end)

    // Each process will compute local_scores of length count
    double *local_scores = (double*)malloc(count * sizeof(double));
    if (!local_scores) {
        fprintf(stderr, "Rank %d: local_scores allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (long local_i = 0; local_i < count; ++local_i) {
        long i = start + local_i;
        double local_score = 0.0;
        long start_j = (i - WINDOW_SIZE < 0) ? 0 : i - WINDOW_SIZE;
        long end_j = (i + WINDOW_SIZE >= n_particles) ? n_particles : i + WINDOW_SIZE + 1;
        for (long j = start_j; j < end_j; ++j) {
            if (i == j) continue;
            double t_diff = (double)llabs(particles[i].time_min - particles[j].time_min);
            double d_dist = haversine_km(particles[i].lat, particles[i].lon,
                                         particles[j].lat, particles[j].lon);
            double denom = 1.0 + (ALPHA * d_dist) + (BETA * t_diff);
            local_score += (1.0 / denom);
        }
        local_scores[local_i] = local_score;
    }

    // Prepare counts and displacements for gathering results to root
    int *recvcounts = NULL;
    int *displs = NULL;
    if (rank == 0) {
        recvcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
    }
    int mycount = (int)count;
    MPI_Gather(&mycount, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        displs[0] = 0;
        for (int r = 1; r < size; ++r) displs[r] = displs[r-1] + recvcounts[r-1];
    }

    // global_scores only exists on root
    double *global_scores = NULL;
    if (rank == 0) global_scores = (double*)malloc(n_particles * sizeof(double));

    MPI_Gatherv(local_scores, mycount, MPI_DOUBLE,
                global_scores, recvcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    double t1 = MPI_Wtime();
    if (rank == 0) {
        printf("------------------------------------------------\n");
        printf("MPI ranks: %d\n", size);
        printf("Parallel Processing Time (wall): %.6f seconds\n", t1 - t0);
        printf("Total Interactions (approx): ~%.2e\n", (double)n_particles * WINDOW_SIZE * 2);
        printf("------------------------------------------------\n");

        // write output file using global_scores
        FILE *out = fopen(OUTPUT_FILE, "w");
        if (!out) {
            fprintf(stderr, "Cannot open output file %s for writing\n", OUTPUT_FILE);
        } else {
            fprintf(out, "id,lat,lon,time_min,congestion_score\n");
            for (long i = 0; i < n_particles; ++i) {
                // use global_scores[i] as score
                fprintf(out, "%s,%.5f,%.5f,%ld,%.4f\n",
                        particles[i].id, particles[i].lat, particles[i].lon,
                        particles[i].time_min, global_scores[i]);
            }
            fclose(out);
            printf("Rank 0: results saved to %s\n", OUTPUT_FILE);
        }
    }

    // cleanup
    free(local_scores);
    free(particles);
    if (rank == 0) {
        free(global_scores);
        free(recvcounts);
        free(displs);
    }

    MPI_Finalize();
    return 0;
}
