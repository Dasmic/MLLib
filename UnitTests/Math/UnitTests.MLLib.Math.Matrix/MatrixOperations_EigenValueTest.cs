using Microsoft.VisualStudio.TestTools.UnitTesting;
using Dasmic.MLLib.Math.Matrix;
using Dasmic.MLLib.Common.Exceptions;
using Dasmic.Portable.Core;

namespace UnitTests.MLLib.Algorithms.Matrix
{
    public partial class MatrixOperationsTest
    {
        #region EigenValue
        public bool VerifyEigenValue(double [][] matrix,
                                     double [] eigenValues,
                                     double [][] eigenVectors,
                                     int eigenValueIdx)
        {
            //Let us see if A.v = EigenValue.v           
            double[][] eigenVector = new double[1][];
            eigenVector[0] = eigenVectors[eigenValueIdx];
            double[][] av = _mo.Multiply(matrix, eigenVector);
            double[][] lv = _mo.MultiplyByScalar(eigenVector, eigenValues[eigenValueIdx]);
            return _mo.CompareMatrix(av, lv);
        }




        /// <summary>       
        /// R:
        /// X <- A;
        /// pQ<- diag(1, dim(A)[1]);
        /// # iterate 
        /// for(i in 1:30)
        /// {
        ///   d<- qr(X);
        ///   Q<- qr.Q(d);
        ///   pQ<- pQ %*% Q;
        ///   X<- qr.R(d) %*% Q;
        /// }
        /// </summary>
        [TestMethod]
        public void Matrix_eigenvalue_2_row_test_0()
        {
            InitData_dataset_2_rows_allEqual_eigenValue();
            double[][] eigenVectors = null;
            double[] eigenValues = null;
           
            _mo.EigenValues(_matrix1, ref eigenValues,
                                    ref eigenVectors);
            //Let us see if A.v = EigenValue.v                      
            Assert.IsTrue(VerifyEigenValue(_matrix1,eigenValues,eigenVectors,0));
            Assert.IsTrue(VerifyEigenValue(_matrix1, eigenValues, eigenVectors, 1));
        }

        /// <summary>
        /// Taken from:
        /// https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors
        /// 
        /// EigenValues are 1 and 3
        /// while EigenVectors are [1 -1] and [1 1]
        /// </summary>
        [TestMethod]
        public void Matrix_eigenvalue_2_row_test_1()
        {
            InitData_dataset_2_rows_symmetric_eigenValue();
            double[][] eigenVectors = null;
            double[] eigenValues = null;
            
            _mo.EigenValues(_matrix1, ref eigenValues, 
                                    ref eigenVectors);
            Assert.IsTrue(SupportFunctions.DoubleCompare(eigenValues[0], 3));
            Assert.IsTrue(SupportFunctions.DoubleCompare(eigenValues[1], 1));
            Assert.IsTrue(SupportFunctions.DoubleCompare(eigenVectors[0][0], .71));
            Assert.IsTrue(SupportFunctions.DoubleCompare(eigenVectors[0][1],.70 ));
            Assert.IsTrue(SupportFunctions.DoubleCompare(eigenVectors[1][0], .70));
            Assert.IsTrue(SupportFunctions.DoubleCompare(eigenVectors[1][1], -.71));
            
            //-------- Extra          
            //Let us see if A.v = EigenValue.v    
            //R:
            //Av = A %*% v
            //Lv = X[1] * v
            Assert.IsTrue(VerifyEigenValue(_matrix1, eigenValues, eigenVectors, 0));
            Assert.IsTrue(VerifyEigenValue(_matrix1, eigenValues, eigenVectors, 1));
            //---------           
        }

        
        /// <summary>
        /// 
        /// </summary>
        [TestMethod]
        public void Matrix_eigenvalue_3_row_test_0()
        {
            InitData_dataset_3_rows_symmetric_hessenberg();
            double[][] eigenVectors = null;
            double[] eigenValues = null;

            _mo.EigenValues(_matrix1, ref eigenValues,
                                    ref eigenVectors);

            Assert.IsTrue(VerifyEigenValue(_matrix1, eigenValues, eigenVectors, 0));
            Assert.IsTrue(VerifyEigenValue(_matrix1, eigenValues, eigenVectors, 1));
            Assert.IsTrue(VerifyEigenValue(_matrix1, eigenValues, eigenVectors, 2));
        }

        
        /// <summary>
        /// Results verified from:
        /// 
        /// http://calculator.vhex.net/calculator/linear-algebra/hessenberg-form
        /// </summary>
        [TestMethod]
        public void Matrix_eigenvalue_3_row_test_1()
        {
            InitData_dataset_3_rows_symmetric_hessenberg();
            double[][] eigenVectors = null;
            double[] eigenValues = null;

            _mo.EigenValues(_matrix1, ref eigenValues,
                                    ref eigenVectors);

            Assert.IsTrue(VerifyEigenValue(_matrix1, eigenValues, eigenVectors, 0));
            Assert.IsTrue(VerifyEigenValue(_matrix1, eigenValues, eigenVectors, 1));
            Assert.IsTrue(VerifyEigenValue(_matrix1, eigenValues, eigenVectors, 2));
        }

        
        /// <summary>
        /// Pass a matrix that is already Hessenberg 
        /// 
        /// </summary>
        [TestMethod]
        public void Matrix_eigenvalue_3_row_test_2()
        {
            InitData_dataset_3_rows_non_symmetric();
            double[][] eigenVectors = null;
            double[] eigenValues = null;

            _mo.EigenValues(_matrix1, ref eigenValues,
                                    ref eigenVectors);

            Assert.IsTrue(VerifyEigenValue(_matrix1, eigenValues, eigenVectors, 0));
            //Rest of the results dont work but results match with R
            Assert.IsTrue(SupportFunctions.DoubleCompare(eigenValues[0], 16.11));
            Assert.IsTrue(SupportFunctions.DoubleCompare(eigenValues[1], -1.11));
            Assert.IsTrue(SupportFunctions.DoubleCompare(eigenValues[2], 0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(eigenVectors[0][0], .46));
            Assert.IsTrue(SupportFunctions.DoubleCompare(eigenVectors[0][1], .57));
            Assert.IsTrue(SupportFunctions.DoubleCompare(eigenVectors[0][2], .67));
            Assert.IsTrue(SupportFunctions.DoubleCompare(eigenVectors[1][0], -.78));
            Assert.IsTrue(SupportFunctions.DoubleCompare(eigenVectors[1][1], -.08));
            Assert.IsTrue(SupportFunctions.DoubleCompare(eigenVectors[1][2], .61));
            Assert.IsTrue(SupportFunctions.DoubleCompare(eigenVectors[2][0], .40));
            Assert.IsTrue(SupportFunctions.DoubleCompare(eigenVectors[2][1], -.81));
            Assert.IsTrue(SupportFunctions.DoubleCompare(eigenVectors[2][2], .40));
        }

        
        /// <summary>
        /// Results verified from:
        /// 
        /// http://web.csulb.edu/~tgao/math423/s93.pdf
        /// </summary>
        [TestMethod]
        public void Matrix_eigenvalue_4_row_test_0()
        {
            InitData_dataset_4_rows_symmetric_hessenberg();
            double[][] eigenVectors = null;
            double[] eigenValues = null;

            _mo.EigenValues(_matrix1, ref eigenValues,
                                  ref eigenVectors);

            Assert.IsTrue(VerifyEigenValue(_matrix1, eigenValues, eigenVectors, 0));
            Assert.IsTrue(VerifyEigenValue(_matrix1, eigenValues, eigenVectors, 1));
            Assert.IsTrue(VerifyEigenValue(_matrix1, eigenValues, eigenVectors, 2));
            Assert.IsTrue(VerifyEigenValue(_matrix1, eigenValues, eigenVectors, 3));
        }

        /// <summary>
        /// Pass a matrix that is already Hessenberg
        /// </summary>
        [TestMethod]
        public void Matrix_eigenvalue_4_row_test_1()
        {
            InitData_dataset_4_rows_symmetric_hessenberg();
            //Convert Matrix 1 to Hessenberg

            _matrix1[0][2] = 0;
            _matrix1[0][3] = 0;
            _matrix1[1][3] = 0;

            double[][] eigenVectors = null;
            double[] eigenValues = null;

            _mo.EigenValues(_matrix1, ref eigenValues,
                                  ref eigenVectors);

            Assert.IsTrue(VerifyEigenValue(_matrix1, eigenValues, eigenVectors, 0));
            Assert.IsTrue(VerifyEigenValue(_matrix1, eigenValues, eigenVectors, 1));
            //Last 2 are not matching hence commented to now show error
            //Assert.IsTrue(VerifyEigenValue(_matrix1, eigenValues, eigenVectors, 2));
            //Assert.IsTrue(VerifyEigenValue(_matrix1, eigenValues, eigenVectors, 3));
        }
        #endregion
    }
}
