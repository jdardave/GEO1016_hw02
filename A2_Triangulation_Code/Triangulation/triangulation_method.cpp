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

bool Triangulation::triangulation(
        float fx, float fy,     /// input: the focal lengths (same for both cameras)
        float cx, float cy,     /// input: the principal point (same for both cameras)
        const std::vector<vec3> &points_0,    /// input: image points (in homogenous coordinates) in the 1st image.
        const std::vector<vec3> &points_1,    /// input: image points (in homogenous coordinates) in the 2nd image.
        std::vector<vec3> &points_3d,         /// output: reconstructed 3D points
        mat3 &R,   /// output: recovered rotation of 2nd camera (used for updating the viewer and visual inspection)
        vec3 &t    /// output: recovered translation of 2nd camera (used for updating the viewer and visual inspection)
) const {

    //--------------------------------------------------------------------------------------------------------------
    // implementation starts ...

    if (points_0.size() != points_1.size() || (points_0.size() < 8) || (points_1.size() < 8)) {
        std::cout << "Invalid Input!!" << std::endl;
        return false;
    }

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

    // Calculate Transformation Matrix
    mat3 Transform0 = S0 * T0;

    // Normalisation
    std::vector<vec3> norm_points_0;
    for (vec3 p:points_0) {
        mat3 new_coord = mat3(Transform0 * p);
        norm_points_0.emplace_back(new_coord[0], new_coord[1], new_coord[2]);
    }

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

    // Calculate Transformation Matrix
    mat3 Transform1 = S1 * T1;

    // Normalisation
    std::vector<vec3> norm_points_1;
    for (vec3 p1:points_1) {
        mat3 new_coord1 = mat3(Transform1 * p1);
        norm_points_1.emplace_back(new_coord1[0], new_coord1[1], new_coord1[2]);
    }

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

    //Compute K matrix (skewness is 0)
    Matrix<double> k_matrix(3, 3, {fx, 0.0, cx,
                                   0.0, fy, cy,
                                   0.0, 0.0, 1});
    mat3 kmat3 = to_mat3(k_matrix);

    //Compute essential matrix E (using K matrix)
    mat3 Emat3 = transpose(kmat3) * F_final * kmat3;
    Matrix<double> E = to_Matrix(Emat3);

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
    assert (determinant(R1) > 0 && determinant(R2) > 0);

    // Find two potential T values (for camera 2)
    std::vector<double> t1 = Ue.get_column(Ve.cols()-1), t2 = -Ue.get_column(Ve.cols()-1);
    //t equals to the last column of Ue

    // Reconstruct 3D points
    // CAMERA 1
    // Projection matrix from K, R and t
    Matrix<double> Rt(3, 4, {1, 0, 0, 0,
                             0, 1, 0, 0,
                             0, 0, 1, 0});
    Matrix<double> M = k_matrix * Rt;

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
        std::vector<double> A1_1_hom = hom_coordinates(A1_1);
        std::vector<double> A1_2_hom = hom_coordinates(A1_2);
        std::vector<double> A2_1_hom = hom_coordinates(A2_1);
        std::vector<double> A2_2_hom = hom_coordinates(A2_2);
        Matrix<double> coordRt1_1(4,1, A1_1_hom);
        coordRt1_1 = R1_t1 * coordRt1_1;
        Matrix<double> coordRt1_2(4,1, A1_2_hom);
        coordRt1_2 = R1_t2 * coordRt1_2;
        Matrix<double> coordRt2_1(4,1, A2_1_hom);
        coordRt2_1 = R2_t1 * coordRt2_1;
        Matrix<double> coordRt2_2(4,1, A2_2_hom);
        coordRt2_2 = R2_t2 * coordRt2_2;
        // Counters
        if (A1_1_hom[2] > 0 && coordRt1_1[2][0] > 0) {
            count1_1 ++;
        }
        if (A1_2_hom[2] > 0 && coordRt1_2[2][0] > 0) {
            count1_2 ++;
        }
        if (A2_1_hom[2] > 0 && coordRt2_1[2][0] > 0) {
            count2_1 ++;
        }
        if (A2_2_hom[2] > 0 && coordRt2_2[2][0] > 0) {
            count2_2 ++;
        }
    }

    std::vector<std::pair<int, Matrix<double>>> Rt_bestFit = {{count1_1, M1_1}, {count1_2, M1_2},
                                                              {count2_1, M2_1}, {count2_2, M2_2}};

    int biggest_value = 0;
    for (const auto& element : Rt_bestFit) {
        if (element.first > biggest_value) {
            biggest_value = element.first;
        }
    }

    Matrix<double> correct_M;
    for (auto & i : Rt_bestFit) {
        if (i.first == biggest_value){
            correct_M = i.second;
        }
    }

    for (int id = 0; id < points_0.size(); id++) {
        Matrix<double> A_final = get_A(M, correct_M, points_0[id], points_1[id]);
        std::vector<double> A_hom = hom_coordinates(A_final);
        vec3 coord_3d = {float(A_hom[0]),
                         float(A_hom[1]),
                         float(A_hom[2])};
        points_3d.emplace_back(coord_3d);
    }
    assert(points_3d.size() == points_0.size() && points_3d.size() == points_1.size());

    // Final R and t matrices
    if (count1_1 == biggest_value) {
        R = to_mat3(R1);
        t = {float(t1[0]), float(t1[1]), float(t1[2])};
    }

    else if (count1_2 == biggest_value) {
        R = to_mat3(R1);
        t = {float(t2[0]), float(t2[1]), float(t2[2])};
    }

    else if (count2_1 == biggest_value) {
        R = to_mat3(R2);
        t = {float(t1[0]), float(t1[1]), float(t1[2])};
    }

    else if (count2_2 == biggest_value) {
        R = to_mat3(R2);
        t = {float(t2[0]), float(t2[1]), float(t2[2])};
    }

    // Final assessment

        std::vector<vec3> points_image1;
        std::ofstream output_file("../points_image1_difference.txt");
        if (output_file.is_open()){
            for (int i=0; i < points_3d.size(); i++) {
                float last_value = (kmat3 * points_3d[i])[2];
                float x = (kmat3 * points_3d[i] / last_value)[0] - points_0[i][0],
                y = (kmat3 * points_3d[i]/ last_value)[1] - points_0[i][1],
                z = (kmat3 * points_3d[i]/ last_value)[2]  - points_0[i][2];
                output_file << x << " " << y << " " << z << "\n";
            }
            output_file.close();
        }

    std::ofstream output_file2("../points_image1.txt");
    if (output_file2.is_open()){
        for (int i=0; i < points_3d.size(); i++) {
            float last_value = (kmat3 * points_3d[i])[2];
            float x = (kmat3 * points_3d[i] / last_value)[0],
                    y = (kmat3 * points_3d[i]/ last_value)[1],
                    z = (kmat3 * points_3d[i]/ last_value)[2];
            output_file2 << x << " " << y << " " << z << "\n";
            points_image1.push_back({x,y,z});
        }
        output_file2.close();
    }
    std::ofstream output_file3("../points_image2.txt");
    if (output_file3.is_open()){
        for (int i=0; i < points_3d.size(); i++) {
            float last_value = (kmat3 * (R * points_3d[i]+t))[2];
            float x = (kmat3 * (R * points_3d[i]+t))[0]/last_value,
            y = (kmat3 * (R * points_3d[i]+t))[1]/last_value,
            z = (kmat3 * (R * points_3d[i]+t))[2]/last_value;
            output_file3 << x << " " << y << " " << z << "\n";
        }
        output_file3.close();
    }

    std::ofstream output_file4("../points_image2_difference.txt");
    if (output_file4.is_open()){
        for (int i=0; i < points_3d.size(); i++) {
            float last_value = (kmat3 * (R * points_3d[i]+t))[2];
            float x = (kmat3 * (R * points_3d[i]+t))[0]/last_value- points_1[i][0],
                    y = (kmat3 * (R * points_3d[i]+t))[1]/last_value- points_1[i][1],
                    z = (kmat3 * (R * points_3d[i]+t))[2]/last_value- points_1[i][2];
            output_file4 << x << " " << y << " " << z << "\n";
        }
        output_file4.close();
    }

    if (points_0.size() != points_3d.size() || points_1.size() != points_3d.size()) {
        std::cout << "Invalid!!" << std::endl;
        return false;
    }

    return points_3d.size() > 0;
}