using Microsoft.VisualStudio.TestTools.UnitTesting;
using Dasmic.MLLib.Math.Matrix;
using Dasmic.MLLib.Common.Exceptions;

namespace UnitTests.MLLib.Algorithms.Matrix
{
    public partial class MatrixOperationsTest
    {
        [TestMethod]
        public void Matrix_LU_decomp_2_row_test()
        {
            _matrix1 = new double[2][];
            _matrix1[0] = new double[] { 4, 6 };
            _matrix1[1] = new double[] { 3, 3 };

            double[][] L = null, U = null, P=null;

            _mo.LupMatrix(_matrix1, 
                ref L, ref U,ref P);

            //Check L
            Assert.AreEqual(L.Length, 2);
            Assert.AreEqual(L[0].Length, 2);
            Assert.AreEqual(L[0][0], 6);
            Assert.AreEqual(L[0][1], 4);
            Assert.AreEqual(L[1][0], 0);
            Assert.AreEqual(L[1][1], 1);

            //Check U
            Assert.AreEqual(U.Length, 2);
            Assert.AreEqual(U[0].Length, 2);
            Assert.AreEqual(U[0][0], 1);
            Assert.AreEqual(U[0][1], 0);
            Assert.AreEqual(U[1][0], 0.5);
            Assert.AreEqual(U[1][1], 1);
        }

        [TestMethod]
        public void Matrix_LU_decomp_2_special_row_test()
        {
            _matrix1 = new double[2][];
            _matrix1[0] = new double[] { 0, 1 };
            _matrix1[1] = new double[] { 5, 6 };

            double[][] L = null, U = null, P=null;

            _mo.LupMatrix(_matrix1, ref L, ref U, ref P);

            //Check L
            Assert.AreEqual(L.Length, 2);
            Assert.AreEqual(L[0].Length, 2);
            Assert.AreEqual(L[0][0], 1);
            Assert.AreEqual(L[0][1], 0);
            Assert.AreEqual(L[1][0], 0);
            Assert.AreEqual(L[1][1], 5);

            //Check U
            Assert.AreEqual(U.Length, 2);
            Assert.AreEqual(U[0].Length, 2);
            Assert.AreEqual(U[0][0], 1);
            Assert.AreEqual(U[0][1], 0);
            Assert.AreEqual(U[1][0], 6);
            Assert.AreEqual(U[1][1], 1);
        }


        [TestMethod]
        public void Matrix_LU_decomp_3_row_test()
        {
            _matrix1 = new double[3][];
            _matrix1[0] = new double[] { 4, 2, 2 };
            _matrix1[1] = new double[] { 0, 1, 2 };
            _matrix1[2] = new double[] { 1, 0, 3 };

            double[][] L = null, U = null, P=null;

            _mo.LupMatrix(_matrix1, ref L, ref U, ref P);

            //Check L
            Assert.AreEqual(L.Length, 3);
            Assert.AreEqual(L[0].Length, 3);

            Assert.AreEqual(L[0][0], 4);
            Assert.AreEqual(L[0][1], 2);
            Assert.AreEqual(L[0][2], 2);
            Assert.AreEqual(L[1][0], 0);
            Assert.AreEqual(L[1][1], 2);
            Assert.AreEqual(L[1][2], 1);
            Assert.AreEqual(L[2][0], 0);
            Assert.AreEqual(L[2][1], 0);
            Assert.AreEqual(L[2][2], -1.75);

            //Check U
            Assert.AreEqual(U.Length, 3);
            Assert.AreEqual(U[0].Length, 3);
            Assert.AreEqual(U[0][0], 1);
            Assert.AreEqual(U[0][1], 0);
            Assert.AreEqual(U[0][2], 0);
            Assert.AreEqual(U[1][0], 0);
            Assert.AreEqual(U[1][1], 1);
            Assert.AreEqual(U[1][2], 0);
            Assert.AreEqual(U[2][0], .25);
            Assert.AreEqual(U[2][1], 1.25);
            Assert.AreEqual(U[2][2], 1);
        }


    }
}
