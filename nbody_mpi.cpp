// nbody_mpi.cpp
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define ID_LEN 64
#define WINDOW_SIZE 100  // Reduced for testing
#define ALPHA 1.0
#define BETA 0.5
#define R 6371.0

typedef long long ttime_t;

// ========== FUNCTION PROTOTYPES ==========
double to_radians(double deg);
double haversine_km(double lat1, double lon1, double lat2, double lon2);
int read_input_file_fixed(const char* filename, char*** ids_out, double** lats_out,
    double** lons_out, ttime_t** times_out, long* n_out);

// ========== FUNCTION DEFINITIONS ==========

double to_radians(double deg) {
    return deg * (3.14159265358979323846 / 180.0);
}

double haversine_km(double lat1, double lon1, double lat2, double lon2) {
    double dLat = to_radians(lat2 - lat1);
    double dLon = to_radians(lon2 - lon1);
    double a = sin(dLat / 2.0) * sin(dLat / 2.0) +
        cos(to_radians(lat1)) * cos(to_radians(lat2)) *
        sin(dLon / 2.0) * sin(dLon / 2.0);
    double c = 2.0 * atan2(sqrt(a), sqrt(1.0 - a));
    return R * c;
}
// ========== PARSER  ==========
int read_input_file_fixed(const char* filename, char*** ids_out, double** lats_out,
    double** lons_out, ttime_t** times_out, long* n_out) {

    FILE* fp;
    fopen_s(&fp, filename, "rb");  // Open in binary to handle BOM
    if (!fp) {
        fprintf(stderr, "Error: cannot open input file '%s'\n", filename);
        return 1;
    }

    printf("DEBUG: File opened successfully (size: ");

    // Get file size
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    printf("%ld bytes)\n", file_size);

    // Read first few bytes to check for BOM
    unsigned char bom[3];
    size_t bom_read = fread(bom, 1, 3, fp);

    int has_bom = 0;
    if (bom_read >= 3 && bom[0] == 0xEF && bom[1] == 0xBB && bom[2] == 0xBF) {
        printf("DEBUG: File has UTF-8 BOM\n");
        has_bom = 1;
    }
    else {
        fseek(fp, 0, SEEK_SET);
    }

    // First pass: count lines
    char line[1024];
    long n = 0;
    int line_num = 0;

    printf("DEBUG: Scanning file for data lines...\n");

    while (fgets(line, sizeof(line), fp)) {
        line_num++;
        line[strcspn(line, "\r\n")] = 0;
        if (strlen(line) == 0) continue;

        // Count separators: comma OR tab
        int sep_count = 0;
        for (int i = 0; line[i] != '\0'; i++) {
            if (line[i] == ',' || line[i] == '\t') sep_count++;
        }
        if (sep_count < 3) continue;

        // Detect header (id / ID / Id)
        char* first_sep = strpbrk(line, ",\t");
        if (first_sep) {
            char first_field[100];
            size_t len = first_sep - line;
            strncpy_s(first_field, sizeof(first_field), line, len);
            first_field[len] = '\0';

            if (_stricmp(first_field, "id") == 0) {
                printf("DEBUG: Skipping header line: %s\n", line);
                continue;
            }
        }

        n++;
        if (n <= 3) {
            printf("DEBUG: Sample data line %ld: %s\n", n, line);
        }
    }

    printf("DEBUG: Found %ld data records\n", n);

    if (n == 0) {
        fprintf(stderr, "Error: No valid data records found.\n");
        fclose(fp);
        return 1;
    }

    // Allocate arrays
    char** ids = (char**)malloc(n * sizeof(char*));
    double* lats = (double*)malloc(n * sizeof(double));
    double* lons = (double*)malloc(n * sizeof(double));
    ttime_t* times = (ttime_t*)malloc(n * sizeof(ttime_t));

    if (!ids || !lats || !lons || !times) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(fp);
        free(ids); free(lats); free(lons); free(times);
        return 1;
    }

    // Reset file pointer
    fseek(fp, 0, SEEK_SET);
    if (has_bom) fseek(fp, 3, SEEK_SET);

    long idx = 0;
    line_num = 0;

    printf("DEBUG: Parsing data...\n");

    while (fgets(line, sizeof(line), fp) && idx < n) {
        line_num++;
        line[strcspn(line, "\r\n")] = 0;

        if (strlen(line) == 0) continue;

        // Count separators
        int sep_count = 0;
        for (int i = 0; line[i] != '\0'; i++) {
            if (line[i] == ',' || line[i] == '\t') sep_count++;
        }
        if (sep_count < 3) continue;

        // Header check
        char* first_sep = strpbrk(line, ",\t");
        if (first_sep) {
            char first_field[100];
            size_t len = first_sep - line;
            strncpy_s(first_field, sizeof(first_field), line, len);
            first_field[len] = '\0';
            if (_stricmp(first_field, "id") == 0) continue;
        }

        // Tokenize by comma OR tab
        char* fields[4];
        int field_idx = 0;
        char* context = NULL;
        char* token = strtok_s(line, ",\t", &context);

        while (token && field_idx < 4) {
            fields[field_idx++] = token;
            token = strtok_s(NULL, ",\t", &context);
        }

        if (field_idx == 4) {
            char idbuf[ID_LEN] = { 0 };
            double lat = atof(fields[1]);
            double lon = atof(fields[2]);
            long long t = atoll(fields[3]);

            strncpy_s(idbuf, ID_LEN, fields[0], _TRUNCATE);

            ids[idx] = (char*)malloc(ID_LEN);
            strcpy_s(ids[idx], ID_LEN, idbuf);

            lats[idx] = lat;
            lons[idx] = lon;
            times[idx] = (ttime_t)t;

            if (idx < 3) {
                printf("DEBUG: Parsed record %ld: id='%s', lat=%.6f, lon=%.6f, time=%lld\n",
                    idx, idbuf, lat, lon, t);
            }

            idx++;
        }
    }

    fclose(fp);

    if (idx != n) {
        printf("DEBUG: Adjusting count from %ld to %ld\n", n, idx);
        n = idx;
        ids = (char**)realloc(ids, n * sizeof(char*));
        lats = (double*)realloc(lats, n * sizeof(double));
        lons = (double*)realloc(lons, n * sizeof(double));
        times = (ttime_t*)realloc(times, n * sizeof(ttime_t));
    }

    if (n == 0) {
        fprintf(stderr, "Error: No records could be parsed\n");
        free(ids); free(lats); free(lons); free(times);
        return 1;
    }

    printf("Successfully parsed %ld records\n", n);

    *ids_out = ids;
    *lats_out = lats;
    *lons_out = lons;
    *times_out = times;
    *n_out = n;

    return 0;
}











// ========== MAIN FUNCTION ==========
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) {
            fprintf(stderr, "Usage: mpiexec -n 4 %s input_file.csv\n", argv[0]);
            fprintf(stderr, "Example: mpiexec -n 2 %s taxi_nbody_input_small.csv\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    const char* input_file = argv[1];

    // Only rank 0 reads the file
    long n_total = 0;
    char** all_ids = NULL;
    double* all_lats = NULL;
    double* all_lons = NULL;
    ttime_t* all_times = NULL;

    if (rank == 0) {
        printf("Rank 0: Reading input file '%s'...\n", input_file);
        if (read_input_file_fixed(input_file, &all_ids, &all_lats, &all_lons, &all_times, &n_total)) {
            fprintf(stderr, "Failed to read input file\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        printf("Rank 0: Successfully read %ld particles\n", n_total);

        // Quick validation
        if (n_total > 0) {
            printf("First particle: id=%s, lat=%.6f, lon=%.6f, time=%lld\n",
                all_ids[0], all_lats[0], all_lons[0], (long long)all_times[0]);
            if (n_total > 1) {
                printf("Last particle: id=%s, lat=%.6f, lon=%.6f, time=%lld\n",
                    all_ids[n_total - 1], all_lats[n_total - 1], all_lons[n_total - 1],
                    (long long)all_times[n_total - 1]);
            }
        }
    }

    // Broadcast total count to all ranks
    MPI_Bcast(&n_total, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    if (n_total <= 0) {
        if (rank == 0) fprintf(stderr, "No particles to process\n");
        MPI_Finalize();
        return 1;
    }

    // Allocate memory on non-master ranks
    if (rank != 0) {
        all_lats = (double*)malloc(n_total * sizeof(double));
        all_lons = (double*)malloc(n_total * sizeof(double));
        all_times = (ttime_t*)malloc(n_total * sizeof(ttime_t));
        all_ids = (char**)malloc(n_total * sizeof(char*));

        // Allocate ID buffer as contiguous memory
        char* id_buffer = (char*)malloc(n_total * ID_LEN);
        for (long i = 0; i < n_total; i++) {
            all_ids[i] = id_buffer + (i * ID_LEN);
        }
    }

    // Broadcast data arrays
    MPI_Bcast(all_lats, n_total, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(all_lons, n_total, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(all_times, n_total, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    // Broadcast IDs
    if (rank == 0) {
        char* id_buffer = (char*)malloc(n_total * ID_LEN);
        for (long i = 0; i < n_total; i++) {
            strncpy_s(id_buffer + (i * ID_LEN), ID_LEN, all_ids[i], _TRUNCATE);
        }
        MPI_Bcast(id_buffer, n_total * ID_LEN, MPI_CHAR, 0, MPI_COMM_WORLD);
        free(id_buffer);
    }
    else {
        char* id_buffer = (char*)malloc(n_total * ID_LEN);
        MPI_Bcast(id_buffer, n_total * ID_LEN, MPI_CHAR, 0, MPI_COMM_WORLD);

        // Copy IDs from buffer to array
        for (long i = 0; i < n_total; i++) {
            strncpy_s(all_ids[i], ID_LEN, id_buffer + (i * ID_LEN), _TRUNCATE);
        }
        free(id_buffer);
    }

    // Divide work among ranks
    long particles_per_rank = n_total / size;
    long remainder = n_total % size;

    long start_idx, end_idx;
    if (rank < remainder) {
        start_idx = rank * (particles_per_rank + 1);
        end_idx = start_idx + particles_per_rank + 1;
    }
    else {
        start_idx = rank * particles_per_rank + remainder;
        end_idx = start_idx + particles_per_rank;
    }

    long my_count = end_idx - start_idx;

    if (rank == 0) {
        printf("\nWork distribution across %d MPI ranks:\n", size);
        for (int r = 0; r < size; r++) {
            long s, e;
            if (r < remainder) {
                s = r * (particles_per_rank + 1);
                e = s + particles_per_rank + 1;
            }
            else {
                s = r * particles_per_rank + remainder;
                e = s + particles_per_rank;
            }
            printf("  Rank %d: particles [%ld, %ld) count=%ld\n", r, s, e, e - s);
        }
        printf("\n");
    }

    // Allocate array for my results
    double* my_scores = (double*)malloc(my_count * sizeof(double));
    if (!my_scores) {
        fprintf(stderr, "Rank %d: Failed to allocate scores array\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // Compute congestion scores for my assigned particles
    printf("Rank %d: Computing %ld particles (indices %ld to %ld)...\n",
        rank, my_count, start_idx, end_idx - 1);

    for (long i = 0; i < my_count; i++) {
        long global_idx = start_idx + i;

        // Determine window bounds
        long window_start = global_idx - WINDOW_SIZE;
        if (window_start < 0) window_start = 0;

        long window_end = global_idx + WINDOW_SIZE + 1;
        if (window_end > n_total) window_end = n_total;

        double score = 0.0;
        double lat_i = all_lats[global_idx];
        double lon_i = all_lons[global_idx];
        ttime_t time_i = all_times[global_idx];

        for (long j = window_start; j < window_end; j++) {
            if (j == global_idx) continue;

            double time_diff = (double)llabs((long long)(time_i - all_times[j]));
            double distance = haversine_km(lat_i, lon_i, all_lats[j], all_lons[j]);
            double denominator = 1.0 + (ALPHA * distance) + (BETA * time_diff);
            score += (1.0 / denominator);
        }

        my_scores[i] = score;

        // Progress indicator
        if (i % 1000 == 0 && i > 0) {
            printf("Rank %d: Processed %ld/%ld particles\n", rank, i, my_count);
        }
    }

    double end_time = MPI_Wtime();
    double my_compute_time = end_time - start_time;
    printf("Rank %d: Computation completed in %.2f seconds\n", rank, my_compute_time);

    // Gather all times to rank 0
    double max_time;
    MPI_Reduce(&my_compute_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather all results to rank 0
    double* all_scores = NULL;
    if (rank == 0) {
        all_scores = (double*)malloc(n_total * sizeof(double));
        if (!all_scores) {
            fprintf(stderr, "Rank 0: Failed to allocate final scores array\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Prepare counts and displacements for Gatherv
    int* recv_counts = NULL;
    int* displacements = NULL;

    if (rank == 0) {
        recv_counts = (int*)malloc(size * sizeof(int));
        displacements = (int*)malloc(size * sizeof(int));

        long current_displacement = 0;
        for (int r = 0; r < size; r++) {
            long count;
            if (r < remainder) {
                count = particles_per_rank + 1;
            }
            else {
                count = particles_per_rank;
            }
            recv_counts[r] = (int)count;
            displacements[r] = (int)current_displacement;
            current_displacement += count;
        }
    }

    // Gather results
    MPI_Gatherv(my_scores, (int)my_count, MPI_DOUBLE,
        all_scores, recv_counts, displacements, MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    // Rank 0 writes output file
    if (rank == 0) {
        double total_time = MPI_Wtime() - start_time;
        printf("\n========================================\n");
        printf("MPI Computation Summary:\n");
        printf("  Total particles: %ld\n", n_total);
        printf("  MPI ranks: %d\n", size);
        printf("  Window size: %d\n", WINDOW_SIZE);
        printf("  Max compute time (any rank): %.2f seconds\n", max_time);
        printf("  Total elapsed time: %.2f seconds\n", total_time);
        printf("  Speedup factor: %.2fx\n", max_time > 0 ? (max_time * size) / total_time : 1.0);
        printf("========================================\n");

        // Write output
        FILE* output_file;
        fopen_s(&output_file, "nbody_results.csv", "w");
        if (!output_file) {
            fprintf(stderr, "Failed to open output file for writing\n");
        }
        else {
            fprintf(output_file, "id,lat,lon,time_min,congestion_score\n");
            for (long i = 0; i < n_total; i++) {
                fprintf(output_file, "%s,%.7f,%.7f,%lld,%.8f\n",
                    all_ids[i], all_lats[i], all_lons[i],
                    (long long)all_times[i], all_scores[i]);
            }
            fclose(output_file);
            printf("\nResults written to nbody_results.csv\n");
            printf("First few scores:\n");
            for (long i = 0; i < (n_total < 5 ? n_total : 5); i++) {
                printf("  %s: %.6f\n", all_ids[i], all_scores[i]);
            }
        }

        // Cleanup
        free(all_scores);
        free(recv_counts);
        free(displacements);
    }

    // Cleanup
    free(my_scores);

    if (rank == 0) {
        // Free original allocations
        for (long i = 0; i < n_total; i++) {
            free(all_ids[i]);
        }
    }
    else {
        // Free the single ID buffer on other ranks
        if (all_ids && all_ids[0]) {
            free(all_ids[0]);
        }
    }

    free(all_ids);
    free(all_lats);
    free(all_lons);
    free(all_times);

    MPI_Finalize();
    return 0;
}