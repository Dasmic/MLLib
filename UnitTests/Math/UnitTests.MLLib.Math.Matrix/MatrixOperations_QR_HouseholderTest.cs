using Microsoft.VisualStudio.TestTools.UnitTesting;
using Dasmic.MLLib.Math.Matrix;
using Dasmic.MLLib.Common.Exceptions;
using Dasmic.Portable.Core;

namespace UnitTests.MLLib.Algorithms.Matrix
{
    public partial class MatrixOperationsTest
    {
        #region QR using Householder Reflections

        /// <summary>
        /// Answer validated from:
        /// http://comnuan.com/cmnn0100e/
        /// 
        /// </summary>
        [TestMethod]
        public void Matrix_QR_Householder_2_row_non_symmetric_test_0()
        {
            InitData_dataset_2_rows();
            double[][] Q = null;
            double[][] R = null;

            _mo.QRDecomposition_Householder(_matrix1, ref Q, ref R);

            Assert.IsTrue(SupportFunctions.DoubleCompare(R[0][0], 2.23));
            Assert.IsTrue(SupportFunctions.DoubleCompare(R[0][1], 0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(R[1][0], 4.91));
            Assert.IsTrue(SupportFunctions.DoubleCompare(R[1][1], 0.89));


            Assert.IsTrue(SupportFunctions.DoubleCompare(Q[0][0], .44));
            Assert.IsTrue(SupportFunctions.DoubleCompare(Q[0][1], .89));
            Assert.IsTrue(SupportFunctions.DoubleCompare(Q[1][0], .89));
            Assert.IsTrue(SupportFunctions.DoubleCompare(Q[1][1], -.44));
        }

        /// <summary>
        /// Answer validated from:
        /// http://davidstutz.de/matrix-decompositions/matrix-decompositions/householder/demo
        /// 
        /// </summary>
        [TestMethod]
        public void Matrix_QR_Householder_2_row_non_symmetric_test_1()
        {
            InitData_dataset_2_rows_householder();
            double[][] Q=null;
            double[][] R=null;            

            _mo.QRDecomposition_Householder(_matrix1,ref Q, ref R);

            //Verify R is upper triangular
            Assert.IsTrue(_mo.IsUpperTriangular(R));

            //---Veriy if A = Q.R            
            double[][] T = _mo.Multiply(Q, R);
            Assert.IsTrue(_mo.CompareMatrix(_matrix1, T));

            Assert.IsTrue(SupportFunctions.DoubleCompare(R[0][0], 5));
            Assert.IsTrue(SupportFunctions.DoubleCompare(R[0][1], 0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(R[1][0], 2.4));
            Assert.IsTrue(SupportFunctions.DoubleCompare(R[1][1], 3.2));


            Assert.IsTrue(SupportFunctions.DoubleCompare(Q[0][0], .6));
            Assert.IsTrue(SupportFunctions.DoubleCompare(Q[0][1], .8));
            Assert.IsTrue(SupportFunctions.DoubleCompare(Q[1][0], .8));
            Assert.IsTrue(SupportFunctions.DoubleCompare(Q[1][1], -.6));
        }


        /// <summary>
        /// Answer validated from: R
        /// </summary>
        [TestMethod]
        public void Matrix_QR_Householder_2_row_symmetric_test_2()
        {
            InitData_dataset_2_rows_symmetric_eigenValue();
            double[][] Q = null;
            double[][] R = null;
            _mo.QRDecomposition_Householder(_matrix1, ref Q, ref R);

            //Verify R is upper triangular
            Assert.IsTrue(_mo.IsUpperTriangular(R));

            //---Veriy if A = Q.R            
            double[][] T = _mo.Multiply(Q, R);
            Assert.IsTrue(_mo.CompareMatrix(_matrix1, T));

            Assert.IsTrue(SupportFunctions.DoubleCompare(R[0][0], 2.23));
            Assert.IsTrue(SupportFunctions.DoubleCompare(R[0][1], 0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(R[1][0], 1.78));
            Assert.IsTrue(SupportFunctions.DoubleCompare(R[1][1], -1.34));


            Assert.IsTrue(SupportFunctions.DoubleCompare(Q[0][0], .89));
            Assert.IsTrue(SupportFunctions.DoubleCompare(Q[0][1], .44));
            Assert.IsTrue(SupportFunctions.DoubleCompare(Q[1][0], .44));
            Assert.IsTrue(SupportFunctions.DoubleCompare(Q[1][1], -.89));
        }

        /// <summary>
        /// Answer validated from:
        /// https://en.wikipedia.org/wiki/Givens_rotation
        /// 
        /// </summary>
        [TestMethod]
        public void Matrix_QR_Householder_3_row_test_0()
        {
            InitData_dataset_3_rows_symmetric_givens_rotation();
            double[][] Q = null;
            double[][] R = null;           

            _mo.QRDecomposition_Householder(_matrix1, ref Q, ref R);

            Assert.IsTrue(SupportFunctions.DoubleCompare(R[0][0], 7.81));
            Assert.IsTrue(SupportFunctions.DoubleCompare(R[0][1], 0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(R[0][2], 0));

            Assert.IsTrue(SupportFunctions.DoubleCompare(R[1][0], 4.48));
            Assert.IsTrue(SupportFunctions.DoubleCompare(R[1][1], 4.68));
            Assert.IsTrue(SupportFunctions.DoubleCompare(R[1][2], 0));

            Assert.IsTrue(SupportFunctions.DoubleCompare(R[2][0], 2.56));
            Assert.IsTrue(SupportFunctions.DoubleCompare(R[2][1], .96));
            Assert.IsTrue(SupportFunctions.DoubleCompare(R[2][2], -4.18));

            Assert.IsTrue(SupportFunctions.DoubleCompare(Q[0][0], .76));
            Assert.IsTrue(SupportFunctions.DoubleCompare(Q[0][1], .64));
            Assert.IsTrue(SupportFunctions.DoubleCompare(Q[0][2], 0));

            Assert.IsTrue(SupportFunctions.DoubleCompare(Q[1][0], .3327));
            Assert.IsTrue(SupportFunctions.DoubleCompare(Q[1][1], -.39));
            Assert.IsTrue(SupportFunctions.DoubleCompare(Q[1][2], .85));

            Assert.IsTrue(SupportFunctions.DoubleCompare(Q[2][0], .54));
            Assert.IsTrue(SupportFunctions.DoubleCompare(Q[2][1], -.65));
            Assert.IsTrue(SupportFunctions.DoubleCompare(Q[2][2], -.51));
        }

        /// <summary>
        /// Results verified from:
        /// 
        /// https://www.wolframalpha.com/input/?i=QR+decomposition+%7B%7B1,0,1%7D,%7B0,1,2%7D,%7B1,2,0%7D%7D
        /// </summary>
        [TestMethod]
        public void Matrix_QR_Householder_3_row_test_1()
        {
            InitData_dataset_3_rows_symmetric_hessenberg();
            double[][] Q = null;
            double[][] R = null;
            
            //double[][] origH = _mo.Hessenberg(_matrix1, ref U);
            _mo.QRDecomposition_Householder(_matrix1, ref Q, ref R);

            //Verify R is upper triangular
            Assert.IsTrue(_mo.IsUpperTriangular(R));            
            
            //---Veriy if A = Q.R           
            double[][] T = _mo.Multiply(Q, R);
            Assert.IsTrue(_mo.CompareMatrix(_matrix1,T));           
        }


        
        /// <summary>
        /// Results verified from:
        /// 
        /// https://www.wolframalpha.com/input/?i=QR+decomposition+%7B%7B1,0,1%7D,%7B0,1,2%7D,%7B1,2,0%7D%7D
        /// </summary>
        [TestMethod]
        public void Matrix_QR_Householder_3_row_test_2()
        {
            InitData_dataset_3_rows_non_symmetric();
            double[][] Q = null;
            double[][] R = null;
          
            //double[][] origH = _mo.Hessenberg(_matrix1, ref U);
            _mo.QRDecomposition_Householder(_matrix1, ref Q, ref R);

            //Verify R is upper triangular
            Assert.IsTrue(_mo.IsUpperTriangular(R));

            //---Veriy if A = Q.R         
            double[][] T = _mo.Multiply(Q, R);
            Assert.IsTrue(_mo.CompareMatrix(_matrix1, T));          
        }

     
        /// <summary>
        /// Results verified from:
        /// 
        /// https://www.wolframalpha.com/input/?i=QR+decomposition+%7B%7B1,0,1%7D,%7B0,1,2%7D,%7B1,2,0%7D%7D
        /// </summary>
        [TestMethod]
        public void Matrix_QR_Householder_4_row_test()
        {
            InitData_dataset_4_rows_symmetric_hessenberg();
            double[][] Q = null;
            double[][] R = null;
           
            _mo.QRDecomposition_Householder(_matrix1, ref Q, ref R);
            
            //Verify R is upper triangular
            Assert.IsTrue(_mo.IsUpperTriangular(R));

            //---Veriy if A = Q.R            
            double[][] T = _mo.Multiply(Q, R);
            Assert.IsTrue(_mo.CompareMatrix(_matrix1, T));
        }

        /// <summary>
        /// Validate if A = QR
        /// </summary>
        [TestMethod]
        public void Matrix_QR_Householder_validate_3_row_test()
        {
            InitData_dataset_3_rows_symmetric_hessenberg();
            double[][] Q = null;
            double[][] R = null;
         
            _mo.QRDecomposition_Householder(_matrix1, ref Q, ref R);

            double[][] matA = _mo.Multiply(Q, R);
            
            //Do actual comparison            
            Assert.IsTrue(_mo.CompareMatrix(matA, _matrix1));
        }

        [TestMethod]
        [ExpectedException(typeof(InvalidMatrixException))]
        public void Matrix_QR_Householder_invalid_data_throws_exception()
        {
            InitData_dataset_invalid();
            double[][] Q = null;
            double[][] R = null;
            double[][] U = null;

            _mo.QRDecomposition_Hessenberg(_matrix1, ref Q, ref R, ref U);
        }

        #endregion





    }
}
