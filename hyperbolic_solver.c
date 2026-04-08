#include <stdio.h>
#include <math.h>

#define NUM_MICS 4
#define MAX_ITER 20
#define C 343.0f   // speed of sound

typedef struct {
    float x, y, z;
} Vec3;

// Compute distance
float dist(Vec3 a, Vec3 b) {
    return sqrtf(
        (a.x - b.x)*(a.x - b.x) +
        (a.y - b.y)*(a.y - b.y) +
        (a.z - b.z)*(a.z - b.z)
    );
}

// Solve 3x3 system using Cramer's rule
int solve_3x3(float A[3][3], float b[3], Vec3 *x) {
    float det =
        A[0][0]*(A[1][1]*A[2][2] - A[1][2]*A[2][1]) -
        A[0][1]*(A[1][0]*A[2][2] - A[1][2]*A[2][0]) +
        A[0][2]*(A[1][0]*A[2][1] - A[1][1]*A[2][0]);

    if (fabs(det) < 1e-6) return -1;

    float inv_det = 1.0f / det;

    float dx =
        b[0]*(A[1][1]*A[2][2] - A[1][2]*A[2][1]) -
        A[0][1]*(b[1]*A[2][2] - A[1][2]*b[2]) +
        A[0][2]*(b[1]*A[2][1] - A[1][1]*b[2]);

    float dy =
        A[0][0]*(b[1]*A[2][2] - A[1][2]*b[2]) -
        b[0]*(A[1][0]*A[2][2] - A[1][2]*A[2][0]) +
        A[0][2]*(A[1][0]*b[2] - b[1]*A[2][0]);

    float dz =
        A[0][0]*(A[1][1]*b[2] - b[1]*A[2][1]) -
        A[0][1]*(A[1][0]*b[2] - b[1]*A[2][0]) +
        b[0]*(A[1][0]*A[2][1] - A[1][1]*A[2][0]);

    x->x = dx * inv_det;
    x->y = dy * inv_det;
    x->z = dz * inv_det;

    return 0;
}

// Main solver
int estimate_3d_position(
    Vec3 mic[NUM_MICS],
    float tau[NUM_MICS],   // tau[0]=0, tau[i]=t_i - t_0
    Vec3 *pos
) {
    // Initial guess (center of array)
    pos->x = 0.0f;
    pos->y = 0.0f;
    pos->z = 0.0f;

    for (int iter = 0; iter < MAX_ITER; iter++) {
        float J[3][3];   // Jacobian
        float r[3];      // residual

        float d0 = dist(*pos, mic[0]);

        for (int i = 1; i < NUM_MICS; i++) {
            float di = dist(*pos, mic[i]);

            // residual
            r[i-1] = (di - d0) - C * tau[i];

            // avoid division by zero
            if (di < 1e-6 || d0 < 1e-6) return -1;

            // Jacobian
            J[i-1][0] = (pos->x - mic[i].x)/di - (pos->x - mic[0].x)/d0;
            J[i-1][1] = (pos->y - mic[i].y)/di - (pos->y - mic[0].y)/d0;
            J[i-1][2] = (pos->z - mic[i].z)/di - (pos->z - mic[0].z)/d0;
        }

        // Solve J * delta = -r
        float b[3] = {-r[0], -r[1], -r[2]};
        Vec3 delta;

        if (solve_3x3(J, b, &delta) != 0)
            return -1;

        // Update
        pos->x += delta.x;
        pos->y += delta.y;
        pos->z += delta.z;

        // Convergence check
        if (fabs(delta.x) < 1e-4 &&
            fabs(delta.y) < 1e-4 &&
            fabs(delta.z) < 1e-4)
            break;
    }

    return 0;
}

// Example usage
int main() {
    Vec3 mic[4] = {
        {0, 0, 0},
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1}
    };

    float tau[4] = {0.0f, 0.0002f, 0.0001f, 0.00015f};

    Vec3 pos;

    if (estimate_3d_position(mic, tau, &pos) == 0) {
        float dist = sqrtf(pos.x*pos.x + pos.y*pos.y + pos.z*pos.z);

        printf("Position: (%.3f, %.3f, %.3f)\n", pos.x, pos.y, pos.z);
        printf("Distance from origin: %.3f m\n", dist);
    } else {
        printf("Failed to converge\n");
    }

    return 0;
}
