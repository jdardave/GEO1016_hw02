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
) const
{
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
                 "\t    - make sure your code compiles and can reproduce your results without any modification.\n\n" << std::flush;

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
    if (points_0.size()!=points_1.size() || (points_0.size()<8) || (points_1.size()<8)){
        std::cout<< "Invalid Input!!"<<std::endl;
        return false;
    }

    // TODO: Estimate relative pose of two views. This can be subdivided into
    //      - estimate the fundamental matrix F;
    //      - compute the essential matrix E;
    //      - recover rotation R and t.

    // IMAGE 1
    //Find the centroid
    float x0 = 0.0 ,y0 = 0.0;
    for (vec3 p0:points_0){
        x0 += p0[0];
        y0 += p0[1];
    }
    vec3 centroid={x0/points_0.size(),y0/points_0.size(),1};
    mat3 T1= mat3 (1, 0, -centroid[0], 0, 1, -centroid[1], 0,0, 1);
    std::cout << "Translation matrix for normalisation (image 1): " << T1 << std::endl;

    float dist=0;
    float  x0_mean, y0_mean;
    for (vec3 i:points_0){
        x0_mean = i[0] - centroid[0];
        y0_mean = i[1] - centroid[1];
        dist = dist + ((x0_mean*x0_mean) + (y0_mean*y0_mean));
    }
    float s0 = (sqrt(2) / sqrt(dist));
    mat3 S1 = mat3 (s0, 0, 0, 0, s0, 0, 0, 0, 1);
    std::cout << "Scaling matrix for normalisation (image 1): " << S1 << std::endl;
    // Calculate Transformation Matrix
    mat3 Transform1 = S1*T1;

    std::cout << "Transformation matrix for normalisation (image 1): " << Transform1 << std::endl;
    // New coordinates for normalisation of mean

    std::vector<vec3> norm_points_0;
    for (vec3 p:points_0){
        mat3 new_coord = mat3(Transform1*p);
        norm_points_0.emplace_back(new_coord[0],new_coord[1],new_coord[2]);
    }
    std::cout << "Normalized points (image 1): \n " << norm_points_0 << std::endl;


//    // IMAGE 2
    //Find the centroid
    float x1 = 0.0 ,y1 = 0.0;
    for (vec3 p1:points_1){
        x1 += p1[0];
        y1 += p1[1];
    }
    vec3 centroid1={x1/points_1.size(),y1/points_1.size(),1};
    mat3 T2 (1, 0, -centroid1[0], 0, 1, -centroid1[1], 0,0, 1);
    std::cout << "Translation matrix for normalisation (image 2): " << T2 << std::endl;

    float dist1=0;
    float  x1_mean, y1_mean;
    for (vec3 p1:points_1){
        x1_mean = p1[0] - centroid[0];
        y1_mean = p1[1] - centroid[1];
        dist1 = dist1 + ((x1_mean*x1_mean) + (y1_mean*y1_mean));
    }
    float s1 = (sqrt(2) / sqrt(dist1));
    mat3 S2 = mat3 (s1, 0, 0, 0, s1, 0, 0, 0, 1);
    std::cout << "Scaling matrix for normalisation (image 2): " << S2 << std::endl;
    // Calculate Transformation Matrix
    mat3 Transform2 = S2*T2;

    std::cout << "Transformation matrix for normalisation (image 2): " << Transform2 << std::endl;
    // New coordinates for normalisation of mean

    std::vector<vec3> norm_points_1;
    for (vec3 p1:points_1){
        mat3 new_coord1 = mat3(Transform2*p1);
        norm_points_1.emplace_back(new_coord1[0],new_coord1[1],new_coord1[2]);
    }
    std::cout << "Normalized points (image 2): \n " << norm_points_1 << std::endl;

    // Construct W matrix
    Matrix <double> W(points_0.size(),9,1.0);

    for (int i=0;i<norm_points_0.size();i++){
        double u1 = norm_points_0[i][0];
        double v1 = norm_points_0[i][1];
        double u2 = norm_points_1[i][0];
        double v2 = norm_points_1[i][1];
        W.set_row({u1*u2,v1*u2,u2,u1*v2,v1*v2,v2,u1,v1,1},i);

    }
    // SVD OF W;
    Matrix<double> U(W.rows(), W.rows(), 0.0), S(W.rows(), W.cols(), 0.0), V(W.cols(), W.cols(), 0.0);
    svd_decompose(W,U,S,V);
    std::cout<<" Matrix W" << W <<std::endl;

    std::vector<double> f_data =V.get_column(V.cols()-1);
    Matrix<double> f(3,3,f_data);
    mat3 fmat3 = to_mat3(f);

    std::cout<<fmat3<<std::endl;
    //SVD OF f
    Matrix<double> Uf(f.rows(), f.rows(), 0.0), Sf(f.rows(), f.cols(), 0.0), Vf(f.cols(), f.cols(), 0.0);
    svd_decompose(f,Uf,Sf,Vf);

    // Set last value of s to 0
    Sf.set(Sf.rows()-1,Sf.rows()-1,0);
    std::cout<<Sf<<std::endl;

    // Final Fundamental Matrix
    mat3 F=transpose(Transform1)*fmat3*Transform2;
    std::cout<<F<<std::endl;
//    // let's check their mean distance to the "new" origin
//    double im1 = 0, im2 = 0;
//    for (vec3 p:norm_points_0){
//        double dist = sqrt( pow(p[0], 2)
//                    + pow(p[1], 2));
//        im1 += dist;
//    }
//    for (vec3 p:norm_points_0){
//        double dist = sqrt( pow(p[0], 2)
//                            + pow(p[1], 2));
//        im2 += dist;
//    }
//    double finalMean_im1 = im1 / norm_points_0.size(),
//            finalMean_im2 = im2 / norm_points_1.size();
//    std::cout << "IMAGE 1: Mean before norm: " << mean_im1 << " and mean after norm: " << finalMean_im1 << std::endl;
//    std::cout << "IMAGE 2: Mean before norm: " << mean_im2 << " and mean after norm: " << finalMean_im2 << std::endl;


    // TODO: Reconstruct 3D points. The main task is
    //      - triangulate a pair of image points (i.e., compute the 3D coordinates for each corresponding point pair)

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