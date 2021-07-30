#include "mat.h"

//Matrix
Mat::Mat() { resize(1,1); }
Mat::Mat(size_t rows, size_t cols) { resize(rows, cols); }
Mat::Mat(const Mat &other) { *this = other; }

Mat::Mat(const std::initializer_list<std::initializer_list<double>> &list) { *this = list; }

Mat::Mat(const std::initializer_list<double> &list) { *this = list; }

Mat::Mat(const double d) { *this = d; }

matRow<double>& Mat::operator[](const size_t pos) { return mat[pos]; }

double& Mat::operator>>(const size_t pos) { //faster access to element of row/column vector
    if (rows() == 1) { //row vector
        return mat.at(0).at(pos);
    }
    else if (cols() == 1) { //col vector
        return mat.at(pos).at(0);
    }
    else throw std::runtime_error(">> only works with row/col vectors! You should be using [row][col] for matrices!");
}

const Mat Mat::operator[](const range r) const { //access range of rows of matrix, see range class at top
    Mat retval;
    if (r.row_or_col == C) {
        retval.resize(rows(), r.length());
        for (size_t i = 1; i <= retval.rows(); i++) {
            for (size_t j = 1; j <= retval.cols(); j++) {
                retval[i][j] = (*this)(i)[j+r.start()-1];
            }
        }
    }
    else {
        retval.resize(r.length(), cols());
        for (size_t i = 1; i <= retval.rows(); i++) {
            for (size_t j = 1; j <= retval.cols(); j++) {
                retval[i][j] = (*this)(i+r.start()-1)[j];
            }
        }
    }
    return retval;
}

const matRow<double>& Mat::operator()(const size_t pos) const { return mat[pos]; }

const Mat& Mat::operator=(const Mat &other) { mat = other.mat; return *this; }

const Mat& Mat::operator=(const std::initializer_list<std::initializer_list<double>> &list) { 
    if (!list.size()) throw std::runtime_error("Invalid size " + std::to_string(list.size()) + 'x' + std::to_string(list.begin()->size()) + "!");
    size_t col_size = list.begin()->size();
    for (size_t i = 0; i < list.size(); i++) {
        if ((list.begin()+i)->size() != col_size) throw std::runtime_error("Matrix has inconsistent column sizes! Expected " + std::to_string(col_size) + ", got " + std::to_string((list.begin()+i)->size()));
    }
    resize(list.size(), 1);
    for (size_t i = 0; i < rows(); i++) {
        mat.at(i) = *(list.begin()+i);
    }
    return *this;
}

const Mat& Mat::operator=(const std::initializer_list<double> &list) {
    if (!list.size()) throw std::runtime_error("Invalid size 1x0!");
    resize(1,1);
    mat.at(0) = list;
    return *this;
}

const Mat& Mat::operator=(const double d) {
    resize(1,1);
    mat.at(0).at(0) = d;
    return *this;
}

size_t Mat::rows() const { return mat.size(); }

size_t Mat::cols() const { return  mat.at(0).size(); }

void Mat::resize(size_t rows, size_t cols) {
    if (!rows || !cols) { throw std::runtime_error("Invalid size " + std::to_string(rows) + 'x' + std::to_string(cols) + "!"); }
    mat.resize(rows, matRow<double>());
    for(size_t i = 0; i < this->rows(); i++) {
        mat.at(i).resize(cols,0);
    }
}

Mat Mat::minor(size_t row, size_t col) const { //get the minor of the matrix
    if (!isMat(*this) || rows() != cols()) throw std::runtime_error("Minor can only be called on square matrices!");
    Mat retval(rows()-1, cols()-1);
    size_t subrowindex = 1;
    for (size_t i = 1; i <= rows(); i++) {
        size_t subcolindex = 1;
        for (size_t j = 1; j <= cols(); j++) {
            if (i == row || j == col) { continue; }
            retval[subrowindex][subcolindex] = (*this)(i)[j];
            if (++subcolindex == retval.rows() + 1) subrowindex++;
        }
    }
    return retval;
}

//Math operators
Mat Mat::operator+(const Mat &m) const {
    if (rows() != m.rows() || cols() != m.cols()) throw std::runtime_error("Attempt to add two vectors of non-matching dimensions: " + std::to_string(rows()) + 'x' + std::to_string(cols()) + " and " + std::to_string(m.rows()) + 'x' + std::to_string(m.cols()));
    Mat retval = m;
    for (size_t i = 0; i < retval.rows(); i++) {
        for (size_t j = 0; j < retval.cols(); j++) {
            retval.mat.at(i).at(j) += (*this).mat.at(i).at(j);
        }
    }
    return retval;
}

Mat Mat::operator+=(const Mat &m) { return *this = *this + m; }

Mat Mat::operator-(const Mat &m) const {
    if (rows() != m.rows() || cols() != m.cols()) throw std::runtime_error("Attempt to subtract two matrices of non-matching dimensions: " + std::to_string(rows()) + 'x' + std::to_string(cols()) + " and " + std::to_string(m.rows()) + 'x' + std::to_string(m.cols()));
    Mat retval = *this;
    for (size_t i = 0; i < retval.rows(); i++) {
        for (size_t j = 0; j < retval.cols(); j++) {
            retval.mat.at(i).at(j) -= m.mat.at(i).at(j);
        }
    }
    return retval;
}

Mat Mat::operator-() const { return *this*-1; }

Mat Mat::operator-=(const Mat &m) { return *this = *this - m; }

Mat Mat::operator*(const Mat &m) const { 
    if (isScal(*this)) { return m*mat.at(0).at(0); } //allow 1x1 to be multiplied to any matrix
    if (isScal(m)) { return *this * m.mat.at(0).at(0); }
    if (cols() != m.rows()) throw std::runtime_error("Attempt to multiply two matrices with invalid dimensions: " + std::to_string(rows()) + 'x' + std::to_string(cols()) + " and " + std::to_string(m.rows()) + 'x' + std::to_string(m.cols()));
    Mat retval(rows(), m.cols());
    for (size_t i = 0; i < rows(); i++) {
        for (size_t j = 0; j < m.cols(); j++) {
            double result = 0;
            for (size_t k = 0; k < cols(); k++) {
                result += (*this).mat.at(i).at(k) * m.mat.at(k).at(j);
            }
            retval.mat.at(i).at(j) = result;
        }
    }
    return retval;
}

Mat Mat::operator*=(const Mat &m) { return *this = *this * m; }

Mat Mat::operator*(const double scalar) const {
    Mat retval = *this;
    for (size_t i = 0; i < rows(); i++) {
        for (size_t j = 0; j < cols(); j++) {
            retval.mat.at(i).at(j) *= scalar;
        }
    }
    return retval;
}

Mat Mat::operator*=(const double scalar) { return *this = *this * scalar; }

Mat Mat::operator/(const double scalar) const {
    Mat retval = *this;
    for (size_t i = 0; i < retval.rows(); i++) {
        for (size_t j = 0; j < retval.cols(); j++) {
            retval.mat.at(i).at(j) /= scalar;
        }
    }
    return retval;
}

Mat Mat::operator/=(const double scalar) { return *this = *this / scalar; }

Mat Mat::operator/(const Mat &m) const { //This only works if m is a 1x1
    if (!isScal(m)) throw std::runtime_error("You can only divide two matrices if the second is a 1x1 (for now)");
    return *this/m.mat.at(0).at(0);
}

Mat Mat::operator/=(const Mat &m) { return *this = *this / m; }

Mat Mat::operator^(double exponent) const {
    Mat retval = *this;
    if (isScal(*this)) {
        retval[1][1] = pow(retval[1][1], exponent);
    }
    else if (isMat(*this)) {
        if (rows() != cols()) throw std::runtime_error("Expected a nxn Mat, got " + std::to_string(rows()) + 'x' + std::to_string(cols()));
        if (exponent == 0) return ident(rows());
        if (exponent < 0.0) { //if exponent is negative, inverse matrix and flip sign
            retval = inv(retval);
            exponent *= -1;
        }
        //double int_part;
        //if (std::modf(exponent, &int_part) == 0.0) { //Add this back when doubles are supported for matrices
        int i_exponent = exponent; //cast to int for now
        for (int i = 1; i < i_exponent; i++) { retval *= *this; }
        //}
    }
    else throw std::runtime_error("^ operator is not supported with matrix of size " + std::to_string(rows()) + 'x' + std::to_string(cols()) + '!');
    return retval;
}

Mat Mat::operator^=(double exponent) { return *this = *this ^ exponent; }

Mat Mat::operator^(const Mat &m) const { //this will only work if m is a 1x1
    if (!isScal(m)) throw std::runtime_error("You can only exponentiate two matricies if the second is a 1x1!");
    return *this^m.mat.at(0).at(0);
}
Mat Mat::operator^=(const Mat &m) { return *this = *this ^ m; }

//Comparison operators
bool Mat::operator==(const Mat &other) {
    if (rows() != other.rows() || cols() != other.cols()) return false;
    for (size_t i = 1; i <= rows(); i++) {
        for (size_t j = 1; j <= cols(); j++) {
            if ((*this)(i)[j] != other(i)[j]) return false;
        }
    }
    return true;
}
bool Mat::operator!=(const Mat &other) { return !(*this == other); }

std::ostream& operator<< (std::ostream &lhs, const Mat &m) {
    lhs << m.rows() << 'x' << m.cols() << " =\n";
    if (m.rows() == 1 && m.cols() == 1) {
        lhs << m(1)[1];
    }
    else if (m.rows() == 1) {
        lhs << '<';
        for (size_t i = 1; i <= m.cols(); i++) {
            lhs << m(1)[i];
            if (i < m.cols()) lhs << ", ";
        }
        lhs << '>';
    }
    else {
        for (size_t i = 1; i <= m.rows(); i++) {
            if (i>1) lhs << '\n';
            lhs << "[\t";
            for (size_t j = 1; j <= m.cols(); j++) {
                if (j>1) lhs << '\t';
                lhs << m(i)[j];
            }
            lhs << "\t]";
        }
    }
    return lhs;
}

//Identification
bool isMat(const Mat &m) { return m.rows() > 1 && m.cols() > 1; }
bool isVec(const Mat &m) { return (m.rows() > 1) ^ (m.cols() > 1); }
bool isScal(const Mat &m) { return m.rows() == 1 && m.cols() == 1; }

//Commutative overloads
Mat operator*(const double scalar, const Mat &m) { return m*scalar; }
Mat operator/(const double scalar, const Mat &m) { return m/scalar; }
Mat operator+(const double d, const Mat &m) { return m+d; }
Mat operator-(const double d, const Mat &m) { return -m+d; }

//Scalar Functions:
double abs(const Mat &m) { //This function will calculate the abs of scalars, magnitude of vectors, and determinants of matrices all in one
    if (!isScal(m)) throw std::runtime_error("Abs can only be called on a scalar!");
    double retval = fabs(m(1)[1]);
    return retval;
}

//Vector Functions:
double mag(const Mat &m) {
    if (!isVec(m)) throw std::runtime_error("Mag can only be called on a vector!");
    double retval = 0;
    if (m.rows() == 1) { //row vector 
        for (size_t i = 1; i <= m.cols(); i++) retval += m(1)[i]*m(1)[i];
        retval = sqrt(retval);
    }
    if (m.cols() == 1) { //column vector
        for (size_t i = 1; i <= m.rows(); i++) retval += m(i)[1]*m(i)[1];
        retval = sqrt(retval);
    }
    return retval;
}

Mat norm(const Mat &v) {
    if (!isVec(v)) throw std::runtime_error("Normalize can only be called on a vector!");
    Mat retval = v;
    double magnitude = mag(retval);
    if (v.rows() == 1) { //row vector
        for (size_t i = 1; i <= retval.cols(); i++) retval[1][i] /= magnitude;
    }
    else if (v.cols() == 1) { //col vector
        for (size_t i = 1; i <= retval.rows(); i++) retval[i][1] /= magnitude;
    }
    return retval;
}

double sCross(const Mat &v1, const Mat &v2) { //scalar cross product
    if (!isVec(v1) || !isVec(v2)) throw std::runtime_error("Non-vector entered for cross product!");
    if (v1.rows() != v2.rows() || v1.cols() != v2.cols()) throw std::runtime_error("Vector dimensions must match in cross product!");
    double retval = 0;
    if (v1.rows() == 1) { //row vectors
        if (v1.cols() == 2) retval = v1(1)[1] * v2(1)[2] - v2(1)[1] * v1(1)[2]; //if 2d vector
        else retval = mag(cross(v1,v2));
    }
    else { //col vectors
        if (v1.rows() == 2) retval = v1(1)[1] * v2(2)[1] - v2(1)[1] * v1(2)[1]; //if 2d vector
        else retval = mag(cross(v1,v2));
    }
    return retval;
}

Mat cross(const Mat &v1, const Mat &v2) {
    if (!isVec(v1) || !isVec(v2)) throw std::runtime_error("Non-vector entered for cross product!");
    if (v1.rows() != v2.rows() || v1.cols() != v2.cols()) throw std::runtime_error("Vector dimensions must match in cross product!");
    if (v1.rows() > 3 || v1.cols() > 3) throw std::runtime_error("Cross product is only defined in 3-space!");
    Mat retval;
    Mat temp1 = v1, temp2 = v2;
    Mat combine(2,3); //Matrix with the two vectors combined
    if (v1.rows() == 1) { //row vectors
        retval.resize(1,3); //result is 3d row vec
        temp1.resize(1,3); //add kcomp to any 2d vecs
        temp2.resize(1,3);
        combine[1] = temp1[1];
        combine[2] = temp2[1];
        for (size_t i = 1; i <= 3; i++) {
            Mat sub(2,2); //each sub will always be 2x2
            for (size_t j = 1; j <= 2; j++) {
                size_t sub_colindex = 1;
                for (size_t k = 1; k <= 3; k++) {
                    if (k == i) continue;
                    sub[j][sub_colindex++] = combine[j][k];
                }
            }
            double new_val = det(sub);
            retval[1][i] = (i == 2 && new_val != 0 ? -1 : 1) * new_val; //middle is negative, but we dont want a -0
        }
    }
    else { //col vectors
        retval.resize(3,1); //result is a 3d col vec
        temp1.resize(3,1); //add kcomp to any 2d vecs
        temp2.resize(3,1);
        combine[1] = transpose(temp1)[1]; //transpose the col vectors so they can fit horizontally in the matrix
        combine[2] = transpose(temp2)[1];
        for (size_t i = 1; i <= 3; i++) {
            Mat sub(2,2); //each sub will always be 2x2
            for (size_t j = 1; j <= 2; j++) {
                size_t sub_colindex = 0;
                for (size_t k = 1; k <= 3; k++) {
                    if (k == i) continue;
                    sub[j][sub_colindex++] = combine[j][k];
                }
            }
            double new_val = det(sub);
            retval[i][1] = (i == 2 && new_val != 0 ? -1 : 1) * new_val; //middle is negative, but we dont want a -0
        }
    }
    return retval;
}

double dot(const Mat &v1, const Mat &v2) {
    if (!isVec(v1) || !isVec(v2)) throw std::runtime_error("Non-vector entered for dot product!");
    if (v1.rows() != v2.rows() || v1.cols() != v2.cols()) throw std::runtime_error("Vector dimensions must match in dot product!");
    double retval = 0;
    if (v1.rows() == 1) { //row vectors
        for (size_t i = 1; i <= v1.cols(); i++) {
            retval += v1(1)[i] * v2(1)[i];
        }
    }
    else { //col vectors
        for (size_t i = 1; i <= v1.rows(); i++) {
            retval += v1(i)[1] * v2(i)[1];
        }
    }
    return retval;
}

Mat transpose(const Mat &m) { //flip dimensions
    Mat retval(m.cols(), m.rows());
    for (size_t i = 1; i <= m.rows(); i++) {
        for (size_t j = 1; j <= m.cols(); j++) {
            retval[j][i] = m(i)[j];
        }
    }
    return retval;
}

Mat ident(const size_t size) { //returns an identity matrix with the equivalent size
    Mat retval(size,size);
    for (size_t i = 1; i <= size; i++) {
        retval[i][i] = 1;
    }
    return retval;
}

Mat zero(const size_t size) { //returns a zero matrix with the equivalent size
    return Mat(size,size);
}


double det(const Mat &m) {
    if (!isMat(m) || m.rows() != m.cols()) throw std::runtime_error("Det can only be called on square matrices!");
    if (m.rows() == 2) { //2x2 matrix
        double retval = m(1)[1]*m(2)[2] - m(2)[1]*m(1)[2];
        return retval;
    }
    double retval = 0;
    //If larger than 2x2, split into minor submatrices and recurse (minor determinants method)
    for (size_t i = 1; i <= m.cols(); i++) {
        retval += (i%2==0 ? -1 : 1) * m(1)[i] * det(m.minor(1,i)); //add determinant of minor from top row scaled by minor coordinate and sign
    }
    return retval;
}

Mat inv(const Mat &m) { //Inverse of matrix
    if (!isMat(m) || m.rows() != m.cols()) throw std::runtime_error("Inverse can only be called on square matrices!");
    double deter = det(m);
    if (deter == 0) throw std::runtime_error("Inverse called on a matrix with a determinant of 0!");
    return adj(m)/deter;
}

Mat adj(const Mat &m) { //Adjoint matrix
    if (!isMat(m) || m.rows() != m.cols()) throw std::runtime_error("Adjoint can only be called on square matrices!");
    return transpose(cof(m));
}

Mat cof(const Mat &m) { //Cofactor matrix
    if (!isMat(m) || m.rows() != m.cols()) throw std::runtime_error("Cofactor can only be called on square matrices!");
    Mat retval(m.rows(),m.cols());
    for (size_t i = 1; i <= m.rows(); i++) {
        for (size_t j = 1; j <= m.cols(); j++) {
            retval[i][j] = det(m.minor(i,j)) * ((i+j) % 2 == 0 ? 1 : -1);
        }
    }
    return retval;
}

void print(const Mat &m) {
    std::cout << m << std::endl;
}

//Mathy functions
Mat exp(const Mat &m) { //matrix exponential
    Mat retval(m.rows(),m.cols()), last;
    unsigned int n = 0;
    do {
        last = retval;
        retval += 1/tgamma(n+1) * (m^n);
        n++;
    } while (retval != last);
    return retval;
}

//Misc functions
double Q_rsqrt( double number  ) //the classic
{
    long i;
    double x2, y;
    const double threehalfs = 1.5F;

    x2 = number * 0.5F;
    y  = number;
    i  = * ( long *  ) &y;                       // evil doubleing point bit level hacking
    i  = 0x5f3759df - ( i >> 1  );               // what the fuck? 
    y  = * ( double *  ) &i;
    y  = y * ( threehalfs - ( x2 * y * y  )  );   // 1st iteration
    return y;
}

Mat randIntMat(size_t rows, size_t cols, int min, int max) {
    Mat retval(rows, cols);
    for (size_t i = 1; i <= rows; i++) {
        for (size_t j = 1; j <= cols; j++) {
            retval[i][j] = (rand() % (max-min + 1)) + min;
        }
    }
    return retval;
}
