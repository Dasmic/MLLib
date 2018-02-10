using Microsoft.VisualStudio.TestTools.UnitTesting;
using Dasmic.MLLib.Math.Matrix;
using Dasmic.MLLib.Common.Exceptions;
using Dasmic.Portable.Core;

namespace UnitTests.MLLib.Algorithms.Matrix
{
    public partial class MatrixOperationsTest
    {
        #region Hessenberg
        [TestMethod]
        public void Matrix_hessenberg_2_row_test()
        {
            InitData_dataset_2_rows();
            double[][] U=null;
            double[][] newMatrix =
                _mo.Hessenberg(_matrix1,ref U,true);

            Assert.AreEqual(newMatrix[0][0], 1);
            Assert.AreEqual(newMatrix[0][1], 2);
            Assert.AreEqual(newMatrix[1][0], 3);
            Assert.AreEqual(newMatrix[1][1], 4);
        }


        /// <summary>
        /// Results verified from:
        /// 
        /// https://math.la.asu.edu/~gardner/QR.pdf
        /// </summary>
        [TestMethod]
        public void Matrix_hessenberg_3_row_symmetric_test()
        {
            InitData_dataset_3_rows_symmetric_hessenberg();
            double[][] U = null;
            double[][] newMatrix =
                _mo.Hessenberg(_matrix1,ref U);          

            Assert.AreEqual(newMatrix[0][0], 1);
            Assert.AreEqual(newMatrix[0][1], 1);
            Assert.AreEqual(newMatrix[0][2], 0);
            Assert.AreEqual(newMatrix[1][0], 1);
            Assert.AreEqual(newMatrix[1][1], 0);
            Assert.AreEqual(newMatrix[1][2], 2);
            Assert.AreEqual(newMatrix[2][0], 0);
            Assert.AreEqual(newMatrix[2][1], 2);
            Assert.AreEqual(newMatrix[2][2], 1);

            //Also check U
            Assert.AreEqual(U[0][0], 1);
            Assert.AreEqual(U[0][1], 0);
            Assert.AreEqual(U[0][2], 0);
            Assert.AreEqual(U[1][0], 0);
            Assert.AreEqual(U[1][1], 0);
            Assert.AreEqual(U[1][2], 1);
            Assert.AreEqual(U[2][0], 0);
            Assert.AreEqual(U[2][1], 1);
            Assert.AreEqual(U[2][2], 0);
        }

        /// <summary>
        /// Results verified from:
        /// 
        /// http://calculator.vhex.net/calculator/linear-algebra/hessenberg-form
        /// </summary>
        [TestMethod]
        public void Matrix_hessenberg_3_row_non_symmetric_test()
        {
            InitData_dataset_3_rows_non_symmetric();
            double[][] U = null;
            double[][] newMatrix =
                _mo.Hessenberg(_matrix1,ref U);
                      

            Assert.AreEqual(newMatrix.Length, 3);
            Assert.AreEqual(newMatrix[0].Length, 3);

            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[0][0], 1));
            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[0][1], 3.6));
            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[0][2], 0));

            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[1][0], 8.04));
            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[1][1], 14.23));
            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[1][2], 1.84));

            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[2][0], -.55));
            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[2][1], -.153));
            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[2][2], -.23));
        }


        /// <summary>
        /// Pass a matrix that is already Hessenberg 
        /// 
        /// </summary>
        [TestMethod]
        public void Matrix_hessenberg_3_row_test_2()
        {
            InitData_dataset_3_rows_symmetric_givens_rotation();
            double[][] U = null;
            double[][] newMatrix =
                _mo.Hessenberg(_matrix1,ref U);

            //U is null as matrix is already householder
            Assert.IsNull(U);

            Assert.AreEqual(newMatrix.Length, 3);
            Assert.AreEqual(newMatrix[0].Length, 3);

            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[0][0], 6));
            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[0][1], 5));
            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[0][2], 0));

            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[1][0], 5));
            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[1][1], 1));
            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[1][2], 4));

            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[2][0], 0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[2][1], 4));
            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[2][2], 3));
        }

        /// <summary>
        /// Results verified from:
        /// 
        /// http://web.csulb.edu/~tgao/math423/s93.pdf
        /// </summary>
        [TestMethod]
        public void Matrix_hessenberg_4_row_test_0()
        {
            InitData_dataset_4_rows_symmetric_hessenberg();
            double[][] U = null;
            double[][] newMatrix =
                _mo.Hessenberg(_matrix1, ref U);


            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[0][0], 4));
            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[0][1], 3));
            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[0][2], 0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[0][3], 0));

            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[1][0], 3));
            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[1][1], 10.0/ 3.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[1][2], 5.0 / 3.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[1][3], 0));

            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[2][0], 0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[2][1], 5.0 / 3.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[2][2], -33.0 / 25.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[2][3], -68.0 / 75.0));

            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[3][0], 0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[3][1], 0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[3][2], -68.0 / 75.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[3][3], 149.0 / 75.0 ));
        }

        /// <summary>
        /// Pass a matrix that is already Hessenberg
        /// </summary>
        [TestMethod]
        public void Matrix_hessenberg_4_row_test_1()
        {
            InitData_dataset_4_rows_symmetric_hessenberg();
            //Convert Matrix 1 to Hessenberg

            _matrix1[0][2] = 0;
            _matrix1[0][3] = 0;
            _matrix1[1][3] = 0;

            double[][] U = null;
            double[][] newMatrix =
                _mo.Hessenberg(_matrix1,ref U);

            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[0][0], 4));
            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[0][1], 1));
            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[0][2], 0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[0][3], 0));
            
            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[1][0], 1));
            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[1][1], 2));
            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[1][2], 0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[1][3], 0));

            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[2][0], -2));
            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[2][1], 0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[2][2], 3));
            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[2][3], -2));

            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[3][0], 2));
            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[3][1], 1));
            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[3][2], -2));
            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[3][3], -1));
        }

        /// <summary>
        /// Pass a matrix that is already Hessenberg
        /// </summary>
        [TestMethod]
        public void Matrix_hessenberg_validate_4_row_test_1()
        {
            InitData_dataset_4_rows_symmetric_hessenberg();
          
            double[][] U = null;
            double[][] newMatrix =
                _mo.Hessenberg(_matrix1, ref U);

            //Proof of Logic
            //U = P1.P2
            //H = P2.P1.A.P1.P2
            //Now P1 = Transpose(P1) and P2 = Transpose(P2)
            //H = Transpose(P2).Transpose(P1).A.P1.P2
            //As per Matrix theorem:
            //  transpose(A.B) = transpose(B).transpose(A)
            //Therefore,
            //  H = Transpose(P1.P2).A.P1.P2
            // If ^ is Inverse
            // Transpose(P1.P2)^.H = Transpose(P1.P2)^.Transpose(P1.P2).A.P1.P2
            // Transpose(P1.P2)^.H = I.A.P1.P2
            // Transpose(P1.P2)^.H.(P1.P2)^ = I.A.P1.P2.(P1.P2)^
            // Transpose(P1.P2)^.H.(P1.P2)^ = I.A.I
            // Transpose(P1.P2)^.H.(P1.P2)^ = A ... Eq(1)
            // Now as per theorem,
            // since P1 and P2 are unitary, P1.P2 is unitary
            // As per theorem if U is unitary, U.tranpose(U) = I
            // => tranpose(U) = U^
            // Hence in Eq(1)
            // => Transpose(P1.P2)^.H.(P1.P2)^ = A
            // => ((P1.P2)^)^.H.transpose(P1.P2) = A
            // => (P1.P2).H.transpose(P1.P2) = A
            // => A = U.H.transpose(U) 

            double[][] H = _mo.Multiply(U, newMatrix);
            H = _mo.Multiply(H, _mo.Transpose(U));
            Assert.IsTrue(_mo.CompareMatrix(_matrix1, H));
        }


        #endregion


        [TestMethod]
        [ExpectedException(typeof(InvalidMatrixException))]
        public void Matrix_hessenberg_invalid_data_throws_exception()
        {
            InitData_dataset_invalid();
            double[][] U = null;
            double[][] newMatrix =
                _mo.Hessenberg(_matrix1, ref U);
        }

    }
}
