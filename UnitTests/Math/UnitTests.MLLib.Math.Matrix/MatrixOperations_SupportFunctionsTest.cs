using Microsoft.VisualStudio.TestTools.UnitTesting;
using Dasmic.MLLib.Math.Matrix;
using Dasmic.MLLib.Common.Exceptions;

namespace UnitTests.MLLib.Algorithms.Matrix
{
    public partial class MatrixOperationsTest
    {     
        [TestMethod]
        public void Matrix_isDiagonal_2_row_test_true()
        {
            double[][] _matrix = new double[2][];
            _matrix[0] = new double[] { 1, 0 };
            _matrix[1] = new double[] { 0, 1 };                        
            Assert.IsTrue(_mo.IsDiagonal(_matrix));            
        }

        [TestMethod]
        public void Matrix_isDiagonal_2_row_test_false()
        {
            double[][] _matrix = new double[2][];
            _matrix[0] = new double[] { 1, 1 };
            _matrix[1] = new double[] { 0, 1 };
            Assert.IsFalse(_mo.IsDiagonal(_matrix));
        }

        [TestMethod]
        public void Matrix_isDiagonal_3_row_test_true()
        {
            double[][] _matrix = new double[3][];
            _matrix[0] = new double[] { 1, 0, 0};
            _matrix[1] = new double[] { 0, 1, 0 };
            _matrix[2] = new double[] { 0, 0, 1 };
            Assert.IsTrue(_mo.IsDiagonal(_matrix));
        }

        [TestMethod]
        public void Matrix_isDiagonal_3_row_test_false()
        {
            double[][] _matrix = new double[3][];
            _matrix[0] = new double[] { 1, 1, 0 };
            _matrix[1] = new double[] { 0, 1, 0 };
            _matrix[2] = new double[] { 0, 0, 1 };
            Assert.IsFalse(_mo.IsDiagonal(_matrix));
        }


    }
}
