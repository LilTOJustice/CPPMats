#pragma once
#include <cmath>
#include <vector>
#include <iostream>
#include <utility>

class Mat;
template<typename T>
class matRow;

enum { R, C };


//usage: m[range(0,2,R)] returns rows 0-2 as a 3x(however many columns) matrix
//can be stacked as many times you want: m[range(5,8,R)][range(0,1,C)][range(0,1,R)];
class range {
    friend Mat;
    size_t s, e;
    bool row_or_col;
    public:
    range(size_t new_s, size_t new_e, bool r_or_c) {
        if (new_e < new_s) throw std::runtime_error("Invalid range entered!");
        s = new_s;
        e = new_e;
        row_or_col = r_or_c;
    }
    size_t start() const { return s; }
    size_t end() const { return e; }
    size_t length() const { return end() - start() + 1; }
};


//Backend classes:
template<typename T>
class matRow { //vector with limited functions
    friend Mat;
    matRow() = default;
    std::vector<T> v;
    inline const T& at(size_t pos) const { return v[pos]; }
    inline T& at(size_t pos) { return v[pos]; }
    inline const size_t size() const { return v.size(); }
    inline size_t size() { return v.size(); }
    inline void resize(size_t size, T t = T()) { v.resize(size, t); }
    inline void operator=(const std::initializer_list<T> &list) { v = list; }
    public:
    inline const T& operator[](size_t pos) const { 
        if (pos < 1 || pos > v.size()) throw std::runtime_error("Attempt to access outside of matrix size!");
        return v.at(pos-1); //1-indexing
    }
    inline T& operator[](size_t pos) { 
        if (pos < 1 || pos > v.size()) throw std::runtime_error("Attempt to access outside of matrix size!");
        return v.at(pos-1); //1-indexing
    }
};

//The reason you're here:
class Mat {
    matRow<matRow<double>> mat;
    public:
    Mat();
    Mat(size_t rows, size_t cols); //create empty matrix
    Mat(const Mat &other); //copy cstor
    Mat(const std::initializer_list<std::initializer_list<double>> &list);
    Mat(const std::initializer_list<double> &list);
    Mat(const double d);
    matRow<double>& operator[](const size_t pos); //edit access
    double& operator>>(const size_t pos); //faster access to element of row/column vector
    const Mat operator[](const range r) const; //access range of rows of matrix, see range class at top
    const matRow<double>& operator()(const size_t pos) const; //read-only access (only use for first dimension)
    const Mat& operator=(const Mat &other);
    const Mat& operator=(const std::initializer_list<std::initializer_list<double>> &list);
    const Mat& operator=(const std::initializer_list<double> &list);
    const Mat& operator=(const double d);
    size_t rows() const;
    size_t cols() const;
    void resize(size_t rows, size_t cols); //resize matrix, filling 0s in new spaces
    Mat minor(size_t row, size_t col) const; //get the minor of the matrix

    //Math operators
    Mat operator+(const Mat &m) const; //Matrix+Matrix
    Mat operator+=(const Mat &m);
    Mat operator-(const Mat &m) const; //Matrix-Matrix
    Mat operator-() const; //-Matrix
    Mat operator-=(const Mat &m); 
    Mat operator*(const Mat &m) const; //Matrix*Matrix
    Mat operator*=(const Mat &m);
    Mat operator*(const double scalar) const; //Matrix*scalar
    Mat operator*=(const double scalar);
    Mat operator/(const double scalar) const; //Matrix/scalar
    Mat operator/=(const double scalar);
    Mat operator/(const Mat &m) const; //This only works if m is a 1x1
    Mat operator/=(const Mat &m);
    Mat operator^(double exponent) const; //doubles will only work on 1x1 scalar mats, if a double is attempted to be used, it will be casted to an int
    Mat operator^=(double exponent);
    Mat operator^(const Mat &m) const; //This only works if m is a 1x1
    Mat operator^=(const Mat &m);

    //Comparison operators
    bool operator==(const Mat &other);
    bool operator!=(const Mat &other);

    friend std::ostream& operator << (std::ostream &lhs, const Mat &m);
};
//Identification
bool isMat(const Mat &m);
bool isVec(const Mat &m);
bool isScal(const Mat &m);

//Commutative overloads
Mat operator*(const double scalar, const Mat &m);
Mat operator/(const double scalar, const Mat &m);
Mat operator+(const double d, const Mat &m);
Mat operator-(const double d, const Mat &m);

//Scalar functions:
double abs(const Mat &m); //abs of scalar

//Vector functions:
double mag(const Mat &m); //mag of vector
Mat norm(const Mat &v);
double dot(const Mat &v1, const Mat &v2);
Mat cross(const Mat &v1, const Mat &v2);
double sCross(const Mat &v1, const Mat &v2); //scalar cross product, optimized for length 2 vectors, otherwise equvalent to abs(cross(v1,v2))

std::ostream& operator<< (std::ostream &lhs, const Mat &rhs);


//Matrix functions: 
Mat transpose(const Mat &v); //Flips dimensions
Mat ident(const size_t size); //returns an identity matrix with the equivalent size
Mat zero(const size_t size); //returns a zero matrix with the equivalent size (this is the same as just calling Mat(size,size))
double det(const Mat &m); //det of matrix
Mat inv(const Mat &m); //Inverse of matrix
Mat adj(const Mat &m); //Adjoint matrix
Mat cof(const Mat &m); //Cofactor matrix
void print(const Mat &m); //same as using cout

//Mathy functions:
Mat exp(const Mat &m); //matrix exponential

//Misc functions:
double Q_rsqrt(double number); //the classic
Mat randIntMat(size_t rows, size_t cols, int min, int max); //generate mat of given size filled with random ints
