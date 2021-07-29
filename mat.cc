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
        for (size_t i = 0; i < retval.rows(); i++) {
            for (size_t j = 0; j < retval.cols(); j++) {
                retval[i][j] = (*this)(i)[j+r.start()];
            }
        }
    }
    else {
        retval.resize(r.length(), cols());
        for (size_t i = 0; i < retval.rows(); i++) {
            for (size_t j = 0; j < retval.cols(); j++) {
                retval[i][j] = (*this)(i+r.start())[j];
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
    Mat retval = m;
    for (size_t i = 0; i < retval.rows(); i++) {
        for (size_t j = 0; j < retval.cols(); j++) {
            retval.mat.at(i).at(j) -= (*this).mat.at(i).at(j);
        }
    }
    return retval;
}

Mat Mat::operator-=(const Mat &m) { return *this = *this - m; }

Mat Mat::operator*(const Mat &m) const { 
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

std::ostream& operator<< (std::ostream &lhs, const Mat &m) {
    lhs << m.rows() << 'x' << m.cols() << " =\n";
    if (m.rows() == 1 && m.cols() == 1) {
        lhs << m(0)[0];
    }
    else if (m.rows() == 1) {
        lhs << '<';
        for (size_t i = 0; i < m.cols(); i++) {
            lhs << m(0)[i];
            if (i < m.cols()-1) lhs << ", ";
        }
        lhs << '>';
    }
    else {
        for (size_t i = 0; i < m.rows(); i++) {
            if (i) lhs << '\n';
            lhs << "[\t";
            for (size_t j = 0; j < m.cols(); j++) {
                if (j) lhs << '\t';
                lhs << m(i)[j];
            }
            lhs << "\t]";
        }
    }
    return lhs;
}

bool isMat(const Mat &m) { return m.rows() > 1 && m.cols() > 1; }
bool isVec(const Mat &m) { return (m.rows() > 1) ^ (m.cols() > 1); }
bool isScal(const Mat &m) { return m.rows() == 1 && m.cols() == 1; }

//Vector Functions:
Mat norm(const Mat &v) {
    Mat retval = v;
    double mag = abs(retval);
    if (v.rows() == 1) { //row vector
        for (size_t i = 0; i < retval.cols(); i++) retval[0][i] /= mag;
    }
    else if (v.cols() == 1) { //col vector
        for (size_t i = 0; i < retval.rows(); i++) retval[i][0] /= mag;
    }
    else throw std::runtime_error("Attempt to normalize a non-vector!");
    return retval;
}

double sCross(const Mat &v1, const Mat &v2) { //scalar cross product
    if (!isVec(v1) || !isVec(v2)) throw std::runtime_error("Non-vector entered for cross product!");
    if (v1.rows() != v2.rows() || v1.cols() != v2.cols()) throw std::runtime_error("Vector dimensions must match in cross product!");
    double retval = 0;
    if (v1.rows() == 1) { //row vectors
        if (v1.cols() == 2) retval = v1(0)[0] * v2(0)[1] - v2(0)[0] * v1(0)[1]; //if 2d vector
        else retval = abs(cross(v1,v2));
    }
    else { //col vectors
        if (v1.rows() == 2) retval = v1(0)[0] * v2(1)[0] - v2(0)[0] * v1(1)[0]; //if 2d vector
        else retval = abs(cross(v1,v2));
    }
    return retval;
}

Mat cross(const Mat &v1, const Mat &v2) { //TODO
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
        combine[0] = temp1[0]; //transpose the col vectors so they can fit horizontally in the matrix
        combine[1] = temp2[0];
        for (size_t i = 0; i < 3; i++) {
            Mat sub(2,2); //each sub will always be 2x2
            for (size_t j = 1; j < 3; j++) {
                size_t sub_colindex = 0;
                for (size_t k = 0; k < 3; k++) {
                    if (k == i) continue;
                    sub[j-1][sub_colindex++] = combine[j-1][k];
                }
            }
            double new_val = abs(sub);
            retval[0][i] = (i == 1 && new_val != 0 ? -1 : 1) * new_val; //middle is negative, but we dont want a -0
        }
    }
    else { //col vectors
        retval.resize(3,1); //result is a 3d col vec
        temp1.resize(3,1); //add kcomp to any 2d vecs
        temp2.resize(3,1);
        combine[0] = transpose(temp1)[0];
        combine[1] = transpose(temp2)[0];
        for (size_t i = 0; i < 3; i++) {
            Mat sub(2,2); //each sub will always be 2x2
            for (size_t j = 1; j < 3; j++) {
                size_t sub_colindex = 0;
                for (size_t k = 0; k < 3; k++) {
                    if (k == i) continue;
                    sub[j-1][sub_colindex++] = combine[j-1][k];
                }
            }
            double new_val = abs(sub);
            retval[i][0] = (i == 1 && new_val != 0 ? -1 : 1) * new_val; //middle is negative, but we dont want a -0
        }
    }
    return retval;
}

double dot(const Mat &v1, const Mat &v2) {
    if (!isVec(v1) || !isVec(v2)) throw std::runtime_error("Non-vector entered for dot product!");
    if (v1.rows() != v2.rows() || v1.cols() != v2.cols()) throw std::runtime_error("Vector dimensions must match in dot product!");
    double retval = 0;
    if (v1.rows() == 1) { //row vectors
        for (size_t i = 0; i < v1.cols(); i++) {
            retval += v1(0)[i] * v2(0)[i];
        }
    }
    else { //col vectors
        for (size_t i = 0; i < v1.rows(); i++) {
            retval += v1(i)[0] * v2(i)[0];
        }
    }
    return retval;
}

Mat transpose(const Mat &m) { //flip dimensions
    Mat retval(m.cols(), m.rows());
    for (size_t i = 0; i < m.rows(); i++) {
        for (size_t j = 0; j < m.cols(); j++) {
            retval[j][i] = m(i)[j];
        }
    }
    return retval;
}

Mat ident(const size_t size) { //returns an identity matrix with the equivalent size
    Mat retval(size,size);
    for (size_t i = 0; i < size; i++) {
        retval[i][i] = 1;
    }
    return retval;
}

double abs(const Mat &m) { //This function will calculate the abs of scalars, magnitude of vectors, and determinants of matrices all in one
    if (m.rows() == 1) { //row vector (magnitude), incidentally calculates scalar absolute value if a 1x1 mat is entered
        double retval = 0;
        for (size_t i = 0; i < m.cols(); i++) retval += m(0)[i]*m(0)[i];
        retval = sqrt(retval);
        return retval;
    }
    if (m.cols() == 1) { //column vector(magnitude), would also calculate scalar abs
        double retval = 0;
        for (size_t i = 0; i < m.rows(); i++) retval += m(i)[0]*m(i)[0];
        retval = sqrt(retval);
        return retval;
    }
    if (m.rows() != m.cols()) throw std::runtime_error("Matrix passed in to determinant should be a square matrix, instead was " + std::to_string(m.rows()) + 'x' + std::to_string(m.cols()));
    if (m.rows() == 2) { //2x2 matrix
        double retval = m(0)[0]*m(1)[1] - m(1)[0]*m(0)[1];
        return retval;
    }
    //If larger than 2x2, split into submatrices and recurse (minor determinants method)
    std::vector<std::pair<double,Mat>> sub_matrices; //store as pairs to remember what to scale determinants by
    for (size_t i = 0; i < m.cols(); i++) { //go throw each column of top row
        Mat sub(m.rows()-1, m.cols()-1); //new matrix will be n-1 x n-1
        for (size_t j = 1; j < m.rows(); j++) { //skip row 0, these will be the determinant scale values
            size_t sub_colindex = 0;
            for (size_t k = 0; k < m.cols(); k++) {
                if (k == i) continue;
                sub[j-1][sub_colindex++] = m(j)[k];
            }
        }
        sub_matrices.push_back({(i%2==0 ? 1 : -1) * m(0)[i], sub}); //dont forget to alternate signs
    }
    double retval = 0;
    for (const std::pair<double, Mat> &p : sub_matrices) retval += p.first * abs(p.second);
    return retval;
}

void print(const Mat &m) {
    std::cout << m << std::endl;
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
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            retval[i][j] = (rand() % (max + 1)) + min;
        }
    }
    return retval;
}
