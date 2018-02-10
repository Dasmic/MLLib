using Microsoft.VisualStudio.TestTools.UnitTesting;
using Dasmic.MLLib.Math.Matrix;
using Dasmic.MLLib.Common.Exceptions;
using Dasmic.Portable.Core;

namespace UnitTests.MLLib.Algorithms.Matrix
{
    public partial class MatrixOperationsTest
    {
        #region Given's Rotation
        /// <summary>
        /// Answer validated from:
        /// https://en.wikipedia.org/wiki/Givens_rotation
        /// 
        /// </summary>
        [TestMethod]
        public void Matrix_givens_rotation_3_row_test()
        {
            InitData_dataset_3_rows_symmetric_givens_rotation();
            double[][] Q = null;
            double[][] R = null;
            double[][] U = null;
            _mo.QRDecomposition_Hessenberg(_matrix1, ref U, ref Q, ref R);

            //Check if _matrix = QR
            double[][] newM = _mo.Multiply(Q, R);
            Assert.IsTrue(_mo.CompareMatrix(_matrix1, newM));

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
        
        #endregion
       
    }
}
