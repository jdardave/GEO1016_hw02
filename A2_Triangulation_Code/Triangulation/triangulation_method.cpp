/**
 * Copyright (C) 2015 by Liangliang Nan (liangliang.nan@gmail.com)
 * https://3d.bk.tudelft.nl/liangliang/
 *
 * This file is part of Easy3D. If it is useful in your research/work,
 * I would be grateful if you show your appreciation by citing it:
 * ------------------------------------------------------------------
 *      Liangliang Nan.
 *      Easy3D: a lightweight, easy-to-use, and efficient C++
 *      library for processing and rendering 3D data. 2018.
 * ------------------------------------------------------------------
 * Easy3D is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License Version 3
 * as published by the Free Software Foundation.
 *
 * Easy3D is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include "triangulation.h"
#include "matrix_algo.h"
#include <easy3d/optimizer/optimizer_lm.h>
# include <cmath>
# include <unordered_map>
# include <map>


using namespace easy3d;


/// convert a 3 by 3 matrix of type 'Matrix<double>' to mat3
mat3 to_mat3(Matrix<double> &M) {
    mat3 result;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j)
            result(i, j) = M(i, j);
    }
    return result;
}

/// convert M of type 'matN' (N can be any positive integer) to type 'Matrix<double>'
template<typename mat>
Matrix<double> to_Matrix(const mat &M) {
    const int num_rows = M.num_rows();
    const int num_cols = M.num_columns();
    Matrix<double> result(num_rows, num_cols);
    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j)
            result(i, j) = M(i, j);
    }
    return result;
}

Matrix<double> get_A (Matrix<double> M_camera1, Matrix<double> M_camera2, vec3 p0, vec3 p1) {
    std::vector<double> A_row1 = (p0[0] * M_camera1.get_row(2)) - M_camera1.get_row(0),
                        A_row2 = (p0[1] * M_camera1.get_row(2)) - M_camera1.get_row(1),
                        A_row3 = (p1[0] * M_camera2.get_row(2)) - M_camera2.get_row(0),
                        A_row4 = (p1[1] * M_camera2.get_row(2)) - M_camera2.get_row(1);
    Matrix<double> A(4, 4, A_row1.data());
    A.set_row(A_row2, 1);
    A.set_row(A_row3, 2);
    A.set_row(A_row4, 3);
    return A;
}

std::vector<double> hom_coordinates (Matrix<double> A) {
    // SVD decomposition
    Matrix<double> Ua(A.rows(), A.rows(), 0.0),
            Sa(A.rows(), A.cols(), 0.0),
            Va(A.cols(), A.cols(), 0.0);
    svd_decompose(A, Ua, Sa, Va);
    // 3D homogeneous coordinates
    double last_value = Va.get_column(Va.cols() - 1)[3];
    std::vector<double> hom_coord = Va.get_column(Va.cols() - 1) / last_value;
    return hom_coord;
};

/**
 * TODO: Finish this function for reconstructing 3D geometry from corresponding image points.
 * @return True on success, otherwise false. On success, the reconstructed 3D points must be written to 'points_3d'.
 */
bool Triangulation::triangulation(
        float fx, float fy,     /// input: the focal lengths (same for both cameras)
        float cx, float cy,     /// input: the principal point (same for both cameras)
        const std::vector<vec3> &points_0,    /// input: image points (in homogenous coordinates) in the 1st image.
        const std::vector<vec3> &points_1,    /// input: image points (in homogenous coordinates) in the 2nd image.
        std::vector<vec3> &points_3d,         /// output: reconstructed 3D points
        mat3 &R,   /// output: recovered rotation of 2nd camera (used for updating the viewer and visual inspection)
        vec3 &t    /// output: recovered translation of 2nd camera (used for updating the viewer and visual inspection)
) const {
    /// NOTE: there might be multiple workflows for reconstructing 3D geometry from corresponding image points.
    ///       This assignment uses the commonly used one explained in our lecture.
    ///       It is advised to define a function for each sub-task. This way you have a clean and well-structured
    ///       implementation, which also makes testing and debugging easier. You can put your other functions above
    ///       triangulation(), or feel free to put them in one or multiple separate files.

    std::cout << "\nTODO: I am going to implement the triangulation() function in the following file:" << std::endl
              << "\t    - triangulation_method.cpp\n\n";

    std::cout << "[Liangliang]:\n"
                 "\tFeel free to use any data structure and function offered by Easy3D, in particular the following two\n"
                 "\tfiles for vectors and matrices:\n"
                 "\t    - easy3d/core/mat.h  Fixed-size matrices and related functions.\n"
                 "\t    - easy3d/core/vec.h  Fixed-size vectors and related functions.\n"
                 "\tFor matrices with unknown sizes (e.g., when handling an unknown number of corresponding points\n"
                 "\tstored in a file, where their sizes can only be known at run time), a dynamic-sized matrix data\n"
                 "\tstructure is necessary. In this case, you can use the templated 'Matrix' class defined in\n"
                 "\t    - Triangulation/matrix.h  Matrices of arbitrary dimensions and related functions.\n"
                 "\tPlease refer to the corresponding header files for more details of these data structures.\n\n"
                 "\tIf you choose to implement the non-linear method for triangulation (optional task). Please refer to\n"
                 "\t'Tutorial_NonlinearLeastSquares/main.cpp' for an example and some explanations. \n\n"
                 "\tIn your final submission, please\n"
                 "\t    - delete ALL unrelated test or debug code and avoid unnecessary output.\n"
                 "\t    - include all the source code (original code framework + your implementation).\n"
                 "\t    - do NOT include the 'build' directory (which contains the intermediate files in a build step).\n"
                 "\t    - make sure your code compiles and can reproduce your results without any modification.\n\n"
              << std::flush;

    /// Easy3D provides fixed-size matrix types, e.g., mat2 (2x2), mat3 (3x3), mat4 (4x4), mat34 (3x4).
    /// To use these matrices, their sizes should be known to you at the compile-time (i.e., when compiling your code).
    /// Once defined, their sizes can NOT be changed.
    /// In 'Triangulation/matrix.h', another templated 'Matrix' type is also provided. This type can have arbitrary
    /// dimensions and their sizes can be specified at run-time (i.e., when executing your program).
    /// Below are a few examples showing some of these data structures and related APIs.

    /// ----------- fixed-size matrices

    /// define a 3 by 4 matrix M (you can also define 3 by 4 matrix similarly)
//    mat34 M(1.0f);  /// entries on the diagonal are initialized to be 1 and others to be 0.
//
//    /// set the first row of M
//    M.set_row(0, vec4(1,1,1,1));    /// vec4 is a 4D vector.
//
//    /// set the second column of M
//    M.set_col(1, vec4(2,2,2,2));
//
//    /// get the 3 rows of M
//    vec4 M1 = M.row(0);
//    vec4 M2 = M.row(1);
//    vec4 M3 = M.row(2);
//
//    /// ----------- fixed-size vectors
//
//    /// how to quickly initialize a std::vector
//    std::vector<double> rows = {0, 1, 2, 3,
//                                4, 5, 6, 7,
//                                8, 9, 10, 11};
//    /// get the '2'-th row of M
//    const vec4 b = M.row(2);    // it assigns the requested row to a new vector b
//
//    /// get the '1'-th column of M
//    const vec3 c = M.col(1);    // it assigns the requested column to a new vector c
//
//    /// modify the element value at row 2 and column 1 (Note the 0-based indices)
//    M(2, 1) = b.x;
//
//    /// apply transformation M on a 3D point p (p is a 3D vector)
//    vec3 p(222, 444, 333);
//    vec3 proj = M * vec4(p, 1.0f);  // use the homogenous coordinates. result is a 3D vector
//
//    /// the length of a vector
//    float len = p.length();
//    /// the squared length of a vector
//    float sqr_len = p.length2();
//
//    /// the dot product of two vectors
//    float dot_prod = dot(p, proj);
//
//    /// the cross product of two vectors
//    vec3 cross_prod = cross(p, proj);
//
//    /// normalize this vector
//    cross_prod.normalize();
//
//    /// a 3 by 3 matrix (all entries are intentionally NOT initialized for efficiency reasons)
//    mat3 F;
//    /// ... here you compute or initialize F.
//    /// compute the inverse of K
//    mat3 invF = inverse(F);
//
//    /// ----------- dynamic-size matrices
//
//    /// define a non-fixed size matrix
//    Matrix<double> W(2, 3, 0.0); // all entries initialized to 0.0.
//
//    /// set its first row by a 3D vector (1.1, 2.2, 3.3)
//    W.set_row({ 1.1, 2.2, 3.3 }, 0);   // here "{ 1.1, 2.2, 3.3 }" is of type 'std::vector<double>'
//
//    /// get the last column of a matrix
//    std::vector<double> last_column = W.get_column(W.cols() - 1);

    // TODO: delete all above demo code in the final submission

    //--------------------------------------------------------------------------------------------------------------
    // implementation starts ...

    // TODO: check if the input is valid (always good because you never known how others will call your function).
    if (points_0.size() != points_1.size() || (points_0.size() < 8) || (points_1.size() < 8)) {
        std::cout << "Invalid Input!!" << std::endl;
        return false;
    }

    // TODO: Estimate relative pose of two views. This can be subdivided into
    //      - estimate the fundamental matrix F;
    //      - compute the essential matrix E;
    //      - recover rotation R and t.

    // IMAGE 1
    //Find the centroid
    float x0 = 0.0, y0 = 0.0;
    for (vec3 p0:points_0) {
        x0 += p0[0];
        y0 += p0[1];
    }
    vec3 centroid = {x0 / points_0.size(), y0 / points_0.size(), 1};
    mat3 T0 = mat3(1, 0, -centroid[0],
                   0, 1, -centroid[1],
                   0, 0, 1);
    std::cout << "Translation matrix for normalisation (image 1): " << T0 << std::endl;

    float dist = 0;
    for (vec3 i:points_0) {
        dist += sqrt(pow((centroid[0]-i[0]), 2)
                + pow((centroid[1]-i[1]), 2));
    }
    float mean_dist0 = dist / points_0.size();
    float s0 = sqrt(2) / mean_dist0;
    mat3 S0 = mat3(s0, 0, 0,
                   0, s0, 0,
                   0, 0, 1);
    std::cout << "Scaling matrix for normalisation (image 1): " << S0 << std::endl;

    // Calculate Transformation Matrix
    mat3 Transform0 = S0 * T0;
    std::cout << "Transformation matrix for normalisation (image 1): " << Transform0 << std::endl;

    // Normalisation
    std::vector<vec3> norm_points_0;
    for (vec3 p:points_0) {
        mat3 new_coord = mat3(Transform0 * p);
        norm_points_0.emplace_back(new_coord[0], new_coord[1], new_coord[2]);
    }
    std::cout << "Normalized points (image 1): \n " << norm_points_0 << std::endl;

    // IMAGE 2
    //Find the centroid
    float x1 = 0.0, y1 = 0.0;
    for (vec3 p1:points_1) {
        x1 += p1[0];
        y1 += p1[1];
    }
    vec3 centroid1 = {x1 / points_1.size(), y1 / points_1.size(), 1};
    mat3 T1(1, 0, -centroid1[0],
            0, 1, -centroid1[1],
            0, 0, 1);
    std::cout << "Translation matrix for normalisation (image 2): " << T1 << std::endl;

    float dist1 = 0;
    for (vec3 p1:points_1) {
        dist1 += sqrt(pow((centroid1[0]-p1[0]), 2)
                     + pow((centroid1[1]-p1[1]), 2));
    }
    float mean_dist1 = dist1 / points_1.size();
    float s1 = sqrt(2) / mean_dist1;
    mat3 S1 = mat3(s1, 0, 0,
                   0, s1, 0,
                   0, 0, 1);
    std::cout << "Scaling matrix for normalisation (image 2): " << S1 << std::endl;

    // Calculate Transformation Matrix
    mat3 Transform1 = S1 * T1;
    std::cout << "Transformation matrix for normalisation (image 2): " << Transform1 << std::endl;

    // Normalisation
    std::vector<vec3> norm_points_1;
    for (vec3 p1:points_1) {
        mat3 new_coord1 = mat3(Transform1 * p1);
        norm_points_1.emplace_back(new_coord1[0], new_coord1[1], new_coord1[2]);
    }
    std::cout << "Normalized points (image 2): \n " << norm_points_1 << std::endl;

    // Construct W matrix
    Matrix<double> W(points_0.size(), 9, 1.0);
    for (int i = 0; i < norm_points_0.size(); i++) {
        double u1 = norm_points_0[i][0],
                v1 = norm_points_0[i][1],
                u2 = norm_points_1[i][0],
                v2 = norm_points_1[i][1];
        W.set_row({u1 * u2, v1 * u2, u2, u1 * v2, v1 * v2, v2, u1, v1, 1},
                  i);
    }
    // SVD OF W;
    Matrix<double> U(W.rows(), W.rows(), 0.0),
            S(W.rows(), W.cols(), 0.0),
            V(W.cols(), W.cols(), 0.0);
    svd_decompose(W, U, S, V);
    //std::cout << "Matrix W " << W << std::endl;

    Matrix<double> f(3, 3, V.get_column(V.cols() - 1));

    //SVD OF f
    Matrix<double> Uf(f.rows(), f.rows(), 0.0),
            Sf(f.rows(), f.cols(), 0.0),
            Vf(f.cols(), f.cols(), 0.0);
    svd_decompose(f, Uf, Sf, Vf);

    // Set last value of s to 0
    Sf.set(Sf.rows() - 1, Sf.cols() - 1, 0);

    // New F matrix
    Matrix<double> f_constraint(3, 3, Uf * Sf * transpose(Vf));
    mat3 fmat3 = to_mat3(f_constraint);

    // Denormalisation
    mat3 F = transpose(Transform1) * fmat3 * Transform0;
    Matrix<double> F_matrix = to_Matrix(F);

    // Scale so that last value of F is 1
    F_matrix /= (F_matrix[F_matrix.rows() - 1][F_matrix.cols() - 1]);
    mat3 F_final = to_mat3(F_matrix); // Final F matrix!!
    std::cout << "Final F " << F_final << std::endl;

    //Compute K matrix (skewness is 0)
    Matrix<double> k_matrix(3, 3, {fx, 0.0, cx,
                                   0.0, fy, cy,
                                   0.0, 0.0, 1});
    mat3 kmat3 = to_mat3(k_matrix);
    std::cout << "K matrix \n" << kmat3 << std::endl;

    //Compute essential matrix E (using K matrix)
    mat3 Emat3 = transpose(kmat3) * F_final * kmat3;
    Matrix<double> E = to_Matrix(Emat3);
    std::cout << "Essential matrix " << E << std::endl;
    // SVD of E
    Matrix<double> Ue(E.rows(), E.rows(), 0.0),
            Se(E.rows(), E.cols(), 0.0),
            Ve(E.cols(), E.cols(), 0.0);
    svd_decompose(E, Ue, Se, Ve);

    // Use helper W and Z matrices to find two values of R (for camera 2)
    Matrix<double> W_matrix(3, 3, {0, -1, 0,
                                   1, 0, 0,
                                   0, 0, 1}),
            Z_matrix(3, 3, {0, 1, 0,
                            -1, 0, 0,
                            0, 0, 0});

    // R values
    Matrix<double> R1 = determinant(Ue * W_matrix * transpose(Ve)) * (Ue * W_matrix * transpose(Ve)),
            R2 = determinant(Ue * transpose(W_matrix) * transpose(Ve)) * (Ue * transpose(W_matrix) * transpose(Ve));
    std::cout << "R1: \n" << R1 << std::endl;
    std::cout << "R2: \n" << R2 << std::endl;
    assert (determinant(R1) > 0 && determinant(R2) > 0);

    // Find two potential T values (for camera 2)
//    Matrix<double> t_helper(3, 1, {0, 0, 1});

    std::vector<double> t1 = Ue.get_column(Ve.cols()-1), t2 = -Ue.get_column(Ve.cols()-1);
    std::cout << "t1: \n" << t1 << std::endl;
    std::cout << "t2: \n" << t2 << std::endl;
    //t equals to the last column of Ue
    std::cout << "Ue: \n" << Ue << std::endl;

    // TODO: Reconstruct 3D points. The main task is
    //      - triangulate a pair of image points (i.e., compute the 3D coordinates for each corresponding point pair)

    // CAMERA 1
    // Projection matrix from K, R and t
    Matrix<double> Rt(3, 4, {1, 0, 0, 0,
                             0, 1, 0, 0,
                             0, 0, 1, 0});
    Matrix<double> M = k_matrix * Rt;
    std::cout << "M first camera \n" << M << std::endl;

    // CAMERA 2
    // 4 Projection matrices for different values of R and t
    Matrix<double> R1_t1(3, 4, {R1[0][0], R1[0][1], R1[0][2], t1[0],
                                        R1[1][0], R1[1][1], R1[1][2], t1[1],
                                        R1[2][0], R1[2][1], R1[2][2], t1[2]}),

            R1_t2(3, 4, {R1[0][0], R1[0][1], R1[0][2], t2[0],
                                     R1[1][0], R1[1][1], R1[1][2], t2[1],
                                     R1[2][0], R1[2][1], R1[2][2], t2[2]}),

            R2_t1(3, 4, {R2[0][0], R2[0][1], R2[0][2], t1[0],
                                     R2[1][0], R2[1][1], R2[1][2], t1[1],
                                     R2[2][0], R2[2][1], R2[2][2], t1[2]}),

            R2_t2(3, 4, {R2[0][0], R2[0][1], R2[0][2], t2[0],
                                     R2[1][0], R2[1][1], R2[1][2], t2[1],
                                     R2[2][0], R2[2][1], R2[2][2], t2[2]});
    Matrix<double> M1_1 = k_matrix * R1_t1,
                    M1_2 = k_matrix * R1_t2,
                    M2_1 = k_matrix * R2_t1,
                    M2_2 = k_matrix * R2_t2;

    // 3D COMPUTATION - LINEAR METHOD
    int count1_1 = 0, count1_2 = 0, count2_1 = 0, count2_2 = 0;
    for (int id = 0; id < points_0.size(); id++) {
        // A matrices for R1, R2, t1, t2 (camera 1)
        Matrix<double> A1_1 = get_A(M, M1_1, points_0[id], points_1[id]),
                        A1_2 = get_A(M, M1_2, points_0[id], points_1[id]),
                        A2_1 = get_A(M, M2_1, points_0[id], points_1[id]),
                        A2_2 = get_A(M, M2_2, points_0[id], points_1[id]);
        // Coordinates camera 2: apply Rt
        Matrix<double> coordRt1_1(1,3, hom_coordinates(A1_1));
        coordRt1_1 *= * R1_t1;
        Matrix<double> coordRt1_2(1,3, hom_coordinates(A1_2));
        coordRt1_2 *= * R1_t2;
        Matrix<double> coordRt2_1(1,3, hom_coordinates(A2_1));
        coordRt2_1 *= * R2_t1;
        Matrix<double> coordRt2_2(1,3, hom_coordinates(A2_2));
        coordRt2_2 *= * R2_t2;

        // Counters
        if (hom_coordinates(A1_1)[2] > 0 && coordRt1_1[0][2] > 0) {
            count1_1 ++;
        }
        if (hom_coordinates(A1_2)[2] > 0 && coordRt1_2[0][2] > 0) {
            count1_2 ++;
        }
        if (hom_coordinates(A2_1)[2] > 0 && coordRt2_1[0][2] > 0) {
            count2_1 ++;
        }
        if (hom_coordinates(A2_2)[2] > 0 && coordRt2_2[0][2] > 0) {
            count2_2 ++;
        }
    }
    std::map <int, Matrix<double>> Rt_bestFit = {{count1_1, M1_1}, {count1_2, M1_2},
                                                 {count2_1, M2_1}, {count2_2, M2_2}};
    int biggest_value = 0;
    for (const auto& [key, value] : Rt_bestFit) {
        if (key > biggest_value) {
            biggest_value = key;
        }
    }
    std::cout << "Final M matrix \n" << Rt_bestFit[biggest_value] << std::endl;

    for (int id = 0; id < points_0.size(); id++) {
        Matrix<double> A_final = get_A(M, Rt_bestFit[biggest_value], points_0[id], points_1[id]);
        vec3 coord_3d = {float(hom_coordinates(A_final)[0]),
                         float(hom_coordinates(A_final)[1]),
                         float(hom_coordinates(A_final)[2])};
        points_3d.emplace_back(coord_3d);
    }
    assert(points_3d.size() == points_0.size() && points_3d.size() == points_1.size());

    // Final R and t matrices
    R = to_mat3(R2);
    t = {float(t1[0]), float(t1[1]), float(t1[2])};

    // TODO: Don't forget to
    //          - write your recovered 3D points into 'points_3d' (the viewer can visualize the 3D points for you);
    //          - write the recovered relative pose into R and t (the view will be updated as seen from the 2nd camera,
    //            which can help you to check if R and t are correct).
    //       You must return either 'true' or 'false' to indicate whether the triangulation was successful (so the
    //       viewer will be notified to visualize the 3D points and update the view).
    //       However, there are a few cases you should return 'false' instead, for example:
    //          - function not implemented yet;
    //          - input not valid (e.g., not enough points, point numbers don't match);
    //          - encountered failure in any step.
    return points_3d.size() > 0;
}