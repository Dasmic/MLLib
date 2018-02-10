using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Dasmic.MLLib.Common.DataManagement;

namespace UnitTests.MLLib.Common.DataManagement
{
    [TestClass]
    public class ArrayManipulationTest:BaseTest
    {
        [TestMethod]
        public void ArrayManipulation_1D_col()
        {
            InitData_dataset_3_rows_non_symmetric();
            double[] newArray = ArrayManipulation.GetColAs1DArray(_trainingData, 2);
            Assert.AreEqual(newArray.Length, 3);
            Assert.AreEqual(newArray[0], 7);
            Assert.AreEqual(newArray[1], 8);
            Assert.AreEqual(newArray[2], 9);
        }

        [TestMethod]
        public void ArrayManipulation_1D_row()
        {
            InitData_dataset_3_rows_non_symmetric();
            double[] newArray = ArrayManipulation.GetRowAs1DArray(_trainingData, 2);
            Assert.AreEqual(newArray.Length, 3);
            Assert.AreEqual(newArray[0], 3);
            Assert.AreEqual(newArray[1], 6);
            Assert.AreEqual(newArray[2], 9);
        }

        #region Size Reduction
        [TestMethod]
        public void ArrayManipulation_2D_double_2_col()
        {
            InitData_dataset_3_rows_non_symmetric();
            double[][] newArray = ArrayManipulation.RemoveSpecificColumn2D(_trainingData, 2);
            Assert.AreEqual(newArray.Length, 2);
            Assert.AreEqual(newArray[0].Length, 3);

            Assert.AreEqual(newArray[0][0], 1);
            Assert.AreEqual(newArray[0][1], 2);
            Assert.AreEqual(newArray[0][2], 3);

            Assert.AreEqual(newArray[1][0], 4);
            Assert.AreEqual(newArray[1][1], 5);
            Assert.AreEqual(newArray[1][2], 6);
        }

        [TestMethod]
        public void ArrayManipulation_2D_double_last_col()
        {
            InitData_dataset_3_rows_non_symmetric();
            double[][] newArray = ArrayManipulation.RemoveLastColumn2D(_trainingData);
            Assert.AreEqual(newArray.Length, 2);
            Assert.AreEqual(newArray[0].Length, 3);

            Assert.AreEqual(newArray[0][0], 1);
            Assert.AreEqual(newArray[0][1], 2);
            Assert.AreEqual(newArray[0][2], 3);

            Assert.AreEqual(newArray[1][0], 4);
            Assert.AreEqual(newArray[1][1], 5);
            Assert.AreEqual(newArray[1][2], 6);
        }

        [TestMethod]
        public void ArrayManipulation_2D_string_last_col()
        {
            InitData_dataset_3_rows_string();
           string[][] newArray = ArrayManipulation.RemoveLastColumn2D(_dataString);

            Assert.AreEqual(newArray.Length, 2);
            Assert.AreEqual(newArray[0].Length, 3);

            Assert.AreEqual(newArray[0][0], "a");
            Assert.AreEqual(newArray[0][1], "b");
            Assert.AreEqual(newArray[0][2], "c");

            Assert.AreEqual(newArray[1][0], "d");
            Assert.AreEqual(newArray[1][1], "e");
            Assert.AreEqual(newArray[1][2], "f");
        }

        [TestMethod]
        public void ArrayManipulation_1D_double_0_col()
        {
            double[] array = new double[] { 2, 3, 4, 5 };
            double[] newArray = ArrayManipulation.RemoveSpecificColumn1D(array,0);
            Assert.AreEqual(newArray.Length, 3);
            
            Assert.AreEqual(newArray[0], 3);
            Assert.AreEqual(newArray[1], 4);
            Assert.AreEqual(newArray[2], 5);           
        }

        [TestMethod]
        public void ArrayManipulation_1D_double_last_col()
        {
            double[] array = new double[] { 2, 3, 4, 5 };
            double[] newArray = ArrayManipulation.RemoveLastColumn1D(array);
            Assert.AreEqual(newArray.Length, 3);

            Assert.AreEqual(newArray[0], 2);
            Assert.AreEqual(newArray[1], 3);
            Assert.AreEqual(newArray[2], 4);
        }

        [TestMethod]
        public void ArrayManipulation_1D_string_0_col()
        {
            string[] array = new string[] { "a", "b", "c", "d" };
            string[] newArray = ArrayManipulation.RemoveSpecificColumn1D(array, 0);
            Assert.AreEqual(newArray.Length, 3);

            Assert.AreEqual(newArray[0], "b");
            Assert.AreEqual(newArray[1], "c");
            Assert.AreEqual(newArray[2], "d");
        }

        [TestMethod]
        public void ArrayManipulation_1D_string_last_col()
        {
            string[] array = new string[] { "a", "b", "c", "d" };
            string[] newArray = ArrayManipulation.RemoveLastColumn1D(array);
            Assert.AreEqual(newArray.Length, 3);

            Assert.AreEqual(newArray[0], "a");
            Assert.AreEqual(newArray[1], "b");
            Assert.AreEqual(newArray[2], "c");
        }

        #endregion Size Reduction
    }
}
