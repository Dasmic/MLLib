using Microsoft.VisualStudio.TestTools.UnitTesting;
using Dasmic.MLLib.Math.Matrix;
using Dasmic.MLLib.Common.Exceptions;

namespace UnitTests.MLLib.Algorithms.Matrix
{
    public partial class MatrixOperationsTest
    {
      
        [TestMethod]
        public void Matrix_scalar_divide_2_row_test()
        {
            InitData_dataset_2_rows();
            double[][] newMatrix =
                _mo.DivideByScalar(_matrix1, 2);

            Assert.AreEqual(newMatrix.Length, 2);
            Assert.AreEqual(newMatrix[0].Length, 2);

            Assert.AreEqual(newMatrix[0][0], .5);
            Assert.AreEqual(newMatrix[0][1], 1);
            Assert.AreEqual(newMatrix[1][0], 1.5);
            Assert.AreEqual(newMatrix[1][1], 2);
        }

        [TestMethod]
        public void Matrix_scalar_divide_nonsquare_row_test()
        {
            InitData_dataset_nonsquare_matrix();
            double[][] newMatrix =
                _mo.DivideByScalar(_matrix1, 2);

            Assert.AreEqual(newMatrix.Length, 2);
            Assert.AreEqual(newMatrix[0].Length, 3);

            Assert.AreEqual(newMatrix[0][0], .5);
            Assert.AreEqual(newMatrix[0][1], 1.5);
            Assert.AreEqual(newMatrix[0][2], 2.5);
            Assert.AreEqual(newMatrix[1][0], 1);
            Assert.AreEqual(newMatrix[1][1], 2);
            Assert.AreEqual(newMatrix[1][2], 3);
        }

        [TestMethod]
        public void Matrix_scalar_multiply_2_row_test()
        {
            InitData_dataset_2_rows();
            double[][] newMatrix =
                _mo.MultiplyByScalar(_matrix1, 2);

            Assert.AreEqual(newMatrix.Length, 2);
            Assert.AreEqual(newMatrix[0].Length, 2);

            Assert.AreEqual(newMatrix[0][0], 2);
            Assert.AreEqual(newMatrix[0][1], 4);
            Assert.AreEqual(newMatrix[1][0], 6);
            Assert.AreEqual(newMatrix[1][1], 8);
        }

        [TestMethod]
        public void Matrix_scalar_multiply_nonsquare_row_test()
        {
            InitData_dataset_nonsquare_matrix();

            double[][] newMatrix =
                _mo.MultiplyByScalar(_matrix1, 2);

            Assert.AreEqual(newMatrix.Length, 2);
            Assert.AreEqual(newMatrix[0].Length, 3);

            Assert.AreEqual(newMatrix[0][0], 2);
            Assert.AreEqual(newMatrix[0][1], 6);
            Assert.AreEqual(newMatrix[0][2], 10);
            Assert.AreEqual(newMatrix[1][0], 4);
            Assert.AreEqual(newMatrix[1][1], 8);
            Assert.AreEqual(newMatrix[1][2], 12);
        }
    }
}
