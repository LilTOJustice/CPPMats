#include "mat.h"
#include <iomanip>

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                         *
 *                                 DEMO.CC                                 *
 *                                                                         *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

//Additional documentation can be found in mat.h

//Everything is a matrix, including scalars and vectors

int main() {
    srand(time(0)); //randIntMat() uses rand() so dont forget to seed
    Mat m = randIntMat(2,2,0,5); //Create 2x2 matrix with random ints from 0 to 5
    Mat n = {{5.3,9.6},{0.1,8.5}}; //2x2 matrix
    Mat iden = ident(7); //Make identity matrix of any size, here's a 7x7
    Mat rowvec = {26,1,3};  //3d row-vector
    Mat colvec = transpose(rowvec); //3d column-vector, transpose flips dimensions
    Mat scalar = 3.14; //Even scalars can be assigned to matrices, this makes a 1x1 matrix
    
    //Now lets print everything:
    std::cout << "m:\n";
    print(m);
    std::cout << "n:\n";
    print(n);
    std::cout << "iden:\n";
    print(iden);
    std::cout << "rowvec:\n";
    print(rowvec);
    std::cout << "colvec:\n";
    print(colvec);
    std::cout << "scalar:\n";
    std::cout << scalar << std::endl; //you can also use ostreams

    //Matrix/Vector math:

    std::cout << "rowvec dot colvec transposed:\n"; //you can also use ostreams
    print(dot(rowvec,transpose(colvec))); //the colvec (or rowvec) needs to be transposed for dot/cross product
    Mat other = {5,2,13};
    
    std::cout << "rowvec x other:\n";
    print(cross(rowvec,other)); //and you can cross them
    
    std::cout << "rowvec x other magnitude:\n";
    print(sCross(rowvec,other)); //scalar cross product
    
    std::cout << "m*n:\n";
    print(m*n); //matrix multiplication, using * on two vectors treats them as 1-d matrices

    m *= 2; //scale matrix
    rowvec *= 3; //scale vector
    scalar *= 5; //scale... scalara
    Mat p = m+n; //add matrices of same dimension

    std::cout << "Determinant of p:\n";
    print(abs(p)); //abs is the det of a matrix, mag of a vector, and abs of a scalar

    //Element access
    std::cout << "modified m:\n";
    m[0][1] = 3; //row, column, 0-indexed
    std::cout << "modified rowvec:\n";
    rowvec>>2 = 57; //Instead of having to specify row and column, for vectors you can just use >>
    
    //Range access: all submatrices from ranges are r-values only.
    Mat largemat = randIntMat(7,8,1,20);
    Mat submatrix = largemat[range(2,5,R)]; //you can also grab a range of rows/cols from a matrix
    std::cout << "submatrix:\n";
    print(submatrix);
    Mat subsubmatrix =  largemat[range(1,6,C)][range(2,4,R)][range(0,1,C)]; //Because each range access returns a Mat, you can stack as many accesses as you want, evaluates left to right
    std::cout << "subsubmatrix:\n";
    print(subsubmatrix); //test and see if you can guess the dimensions of this matrix before it prints
}
