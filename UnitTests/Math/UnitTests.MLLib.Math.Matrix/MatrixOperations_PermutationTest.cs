using Microsoft.VisualStudio.TestTools.UnitTesting;
using Dasmic.MLLib.Math.Matrix;
using Dasmic.MLLib.Common.Exceptions;

namespace UnitTests.MLLib.Algorithms.Matrix
{
    public partial class MatrixOperationsTest
    {
        [TestMethod]
        public void Matrix_permutation_2_row_test()
        {
            _matrix1 = new double[2][];
            _matrix1[0] = new double[] { 0, 1 };
            _matrix1[1] = new double[] { 5, 6 };

            double[][] newMatrix=_mo.PermutationMatrix(_matrix1);
            
            Assert.AreEqual(newMatrix.Length,2);
            Assert.AreEqual(newMatrix[0].Length, 2); 

            Assert.AreEqual(newMatrix[0][0], 0);
            Assert.AreEqual(newMatrix[0][1], 1);
            Assert.AreEqual(newMatrix[1][0], 1);
            Assert.AreEqual(newMatrix[1][1], 0);
        }

        /// <summary>
        /// Example from:
        /// 
        /// http://math.tutorvista.com/algebra/cofactor-matrix.html
        /// </summary>
        [TestMethod]
        public void Matrix_permutation_3_row_test()
        {
            _matrix1 = new double[3][];
            _matrix1[0] = new double[] { 2, -10,-1};
            _matrix1[1] = new double[] { 8, 6, 4};
            _matrix1[2] = new double[] { -9, -4, 7};

            double[][] newMatrix = 
                _mo.PermutationMatrix(_matrix1);

            Assert.AreEqual(newMatrix.Length, 3);
            Assert.AreEqual(newMatrix[0].Length, 3);

            Assert.AreEqual(newMatrix[0][0], 0);
            Assert.AreEqual(newMatrix[0][1], 1);
            Assert.AreEqual(newMatrix[0][2], 0);
            Assert.AreEqual(newMatrix[1][0], 1);
            Assert.AreEqual(newMatrix[1][1], 0);
            Assert.AreEqual(newMatrix[1][2], 0);
            Assert.AreEqual(newMatrix[2][0], 0);
            Assert.AreEqual(newMatrix[2][1], 0);
            Assert.AreEqual(newMatrix[2][2], 1);
        }

        /// <summary>
        /// All 3 rows are different
        /// </summary>
        [TestMethod]
        public void Matrix_permutation_3_row_test_special()
        {
            _matrix1 = new double[3][];
            _matrix1[0] = new double[] { 1, 2, -2 };
            _matrix1[1] = new double[] { 2, 0, 1 };
            _matrix1[2] = new double[] { 2, -1, 3 };

            double[][] newMatrix =
                _mo.PermutationMatrix(_matrix1);

            Assert.AreEqual(newMatrix.Length, 3);
            Assert.AreEqual(newMatrix[0].Length, 3);

            Assert.AreEqual(newMatrix[0][0], 0);
            Assert.AreEqual(newMatrix[0][1], 0);
            Assert.AreEqual(newMatrix[0][2], 1);
            Assert.AreEqual(newMatrix[1][0], 1);
            Assert.AreEqual(newMatrix[1][1], 0);
            Assert.AreEqual(newMatrix[1][2], 0);
            Assert.AreEqual(newMatrix[2][0], 0);
            Assert.AreEqual(newMatrix[2][1], 1);
            Assert.AreEqual(newMatrix[2][2], 0);
        }

    }
}
