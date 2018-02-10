using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Dasmic.MLLib.Math.Statistics;
using Dasmic.Portable.Core;

namespace UnitTests.MLLib.Math.Statistics
{
    [TestClass]
    public class DispersionTest:BaseTest
    {
        double [] ta;
        double[] ta2;

        #region General Tests
        public DispersionTest()
        {
            SetData();
        }

        /// <summary>
        /// R: 
        /// ta<-c(5, 4, 9,14,2,12,26,6, 11)
        /// ta2<-c(8, 10, 12, 14, 18, 21, 24, 26, 29)
        /// </summary>
        private void SetData()
        {
            //9 elements
            ta = new double[] { 5, 4, 9,14,2,12,26,6, 11};
            ta2 = new double[] { 8, 10, 12, 14, 18, 21, 24, 26, 29};
        }

        /// <summary>
        /// R:
        /// sum(ta)
        /// </summary>
        [TestMethod]
        public void Statistics_Sum()
        {
           
            double value = Dispersion.Sum(ta);
            Assert.AreEqual(value, 89);
        }

        /// <summary>
        /// R:
        /// mean(ta)
        /// </summary>
        [TestMethod]
        public void Statistics_Mean()
        {
            double value = Dispersion.Mean(ta);
            Assert.IsTrue((9.887 <= value && value <= 9.889));
        }

        /// <summary>
        /// R:
        /// var(ta)
        /// </summary>
        [TestMethod]
        public void Statistics_VariancePopulation()
        {
            double value = Dispersion.VariancePopulation(ta);
            Assert.IsTrue((46.53 <= value && value <= 46.55));
        }

        /// <summary>
        /// R:
        /// sd(ta)
        /// </summary>
        [TestMethod]
        public void Statistics_Standard_Deviation_Population()
        {
            double value = Dispersion.StandardDeviationPopulation(ta);
            Assert.IsTrue((6.81 <= value && value <= 6.83));
        }

        /// <summary>
        /// R:
        /// sd(ta)
        /// </summary>
        [TestMethod]
        public void Statistics_Standard_Deviation_Population_precomputed_mean()
        {
            double mean = Dispersion.Mean(ta);
            double value = Dispersion.StandardDeviationPopulation(ta,mean);
            Assert.IsTrue((6.81 <= value && value <= 6.83));
        }


        /// <summary>
        /// R:
        /// var(ta)
        /// </summary>
        [TestMethod]
        public void Statistics_VarianceSample()
        {
            double value = Dispersion.VarianceSample(ta);
            Assert.IsTrue((52.360 <= value && value <= 52.362));
        }

        /// <summary>
        /// R:
        /// sd(ta)
        /// </summary>
        [TestMethod]
        public void Statistics_Standard_Deviation_Sample()
        {
            double value = Dispersion.StandardDeviationSample(ta);
            Assert.IsTrue((7.236 <= value && value <= 7.237));
        }

        /// <summary>
        /// R:
        /// NA
        /// 
        /// No number repeats so there should not be any mode
        /// </summary>
        [TestMethod]
        public void Statistics_Mode()
        {
            double value = Dispersion.Mode(ta);
            Assert.AreEqual(Double.NaN, value);
        }

        [TestMethod]
        public void Statistics_Mode_Single_Value()
        {
            double[] na = new double[] { 1 };
            double value = Dispersion.Mode(na);
            Assert.AreEqual(1, value);
        }

        /// <summary>
        /// R:
        /// NA
        /// </summary>
        [TestMethod]
        public void Statistics_Median()
        {
            double value = Dispersion.Median(ta);
            Assert.IsTrue((9 <= value && value <= 9));
        }

        /// <summary>
        /// R:
        /// NA
        /// </summary>
        [TestMethod]
        public void Statistics_Range()
        {
            double value = Dispersion.Range(ta);
            Assert.IsTrue((24 <= value && value <= 24));
        }

        /// <summary>
        /// R:
        /// cor(ta,ta,method="pearson")
        /// </summary>
        [TestMethod]
        public void Statistics_Correlation_One()
        {
            double value = Dispersion.CorrelationPearson(ta, ta);
            Assert.IsTrue(value==1);
        }

        /// <summary>
        /// R:
        /// cor(ta,ta2,method="pearson")
        /// </summary>
        [TestMethod]
        public void Statistics_Correlation_Two()
        {
            double value = Dispersion.CorrelationPearson(ta, ta2);
            Assert.IsTrue((.389 <= value && value <= .391));
        }

        #endregion

        #region CoVariance Tests
        [TestMethod]
        public void Statictics_CoVariance_0()
        {
            double value = Dispersion.CoVarianceSample(ta, ta2);
            Assert.IsTrue(SupportFunctions.DoubleCompare(value,21.125));
        }

        [TestMethod]
        public void Statictics_CoVariance_1()
        {
            InitData_dataset_pca_example();
            double value = Dispersion.CoVarianceSample(_trainingData[0], _trainingData[1]);
            Assert.IsTrue(SupportFunctions.DoubleCompare(value, .6154));
        }

        /// <summary>
        /// R:
        /// m<-cbind(ta,ta2);
        /// cov(m)
        /// </summary>
        [TestMethod]
        public void Statictics_CoVariance_Matrix_2_cols_0()
        {
            double[][] data = new double[2][];
            data[0] = ta;
            data[1] = ta2;
            double[][] matrix = Dispersion.CovarianceMatrixSample(data);
            Assert.IsTrue(SupportFunctions.DoubleCompare(matrix[0][0], 52.36));
            Assert.IsTrue(SupportFunctions.DoubleCompare(matrix[0][1], 21.12));
            Assert.IsTrue(SupportFunctions.DoubleCompare(matrix[1][0], 21.12));
            Assert.IsTrue(SupportFunctions.DoubleCompare(matrix[1][1], 55.75));
        }

        /// <summary>
        /// R:
        ///
        /// cov(m)
        /// </summary>
        [TestMethod]
        public void Statictics_CoVariance_matrix_2_cols_1()
        {
            InitData_dataset_pca_example();
            double[][] matrix = Dispersion.CovarianceMatrixSample(_trainingData);

            Assert.IsTrue(SupportFunctions.DoubleCompare(matrix[0][0], .6165));
            Assert.IsTrue(SupportFunctions.DoubleCompare(matrix[0][1], .6154));
            Assert.IsTrue(SupportFunctions.DoubleCompare(matrix[1][0], .6154));
            Assert.IsTrue(SupportFunctions.DoubleCompare(matrix[1][1], .7165));
        }

        /// <summary>
        /// R:
        /// m<-cbind(ta,ta2,ta2);
        /// cov(m)
        /// </summary>
        [TestMethod]
        public void Statictics_CoVariance_Matrix_3_cols_0()
        {
            double[][] data = new double[3][];
            data[0] = ta;
            data[1] = ta2;
            data[2] = ta2;
            double[][] matrix = Dispersion.CovarianceMatrixSample(data);
            Assert.IsTrue(SupportFunctions.DoubleCompare(matrix[0][0], 52.36));
            Assert.IsTrue(SupportFunctions.DoubleCompare(matrix[0][1], 21.12));
            Assert.IsTrue(SupportFunctions.DoubleCompare(matrix[0][2], 21.12));
            Assert.IsTrue(SupportFunctions.DoubleCompare(matrix[1][0], 21.12));
            Assert.IsTrue(SupportFunctions.DoubleCompare(matrix[1][1], 55.75));
            Assert.IsTrue(SupportFunctions.DoubleCompare(matrix[1][2], 55.75));
            Assert.IsTrue(SupportFunctions.DoubleCompare(matrix[2][0], 21.12));
            Assert.IsTrue(SupportFunctions.DoubleCompare(matrix[2][1], 55.75));
            Assert.IsTrue(SupportFunctions.DoubleCompare(matrix[2][2], 55.75));
        }

        [TestMethod]
        public void Statictics_ZeroCentered_2_rows_0()
        {
            InitData_dataset_pca_example();

            double[][] matrix = Dispersion.GetZeroCenteredData(_trainingData, false);

            Assert.IsTrue(SupportFunctions.DoubleCompare(matrix[0][0], .69));
            Assert.IsTrue(SupportFunctions.DoubleCompare(matrix[0][1], -1.31));
            Assert.IsTrue(SupportFunctions.DoubleCompare(matrix[0][2], .39));
            Assert.IsTrue(SupportFunctions.DoubleCompare(matrix[0][3], .09));
            Assert.IsTrue(SupportFunctions.DoubleCompare(matrix[0][4], 1.29));
            Assert.IsTrue(SupportFunctions.DoubleCompare(matrix[1][0], .49));
            Assert.IsTrue(SupportFunctions.DoubleCompare(matrix[1][1], -1.21));
            Assert.IsTrue(SupportFunctions.DoubleCompare(matrix[1][2], .99));
            Assert.IsTrue(SupportFunctions.DoubleCompare(matrix[1][3], .29));
            Assert.IsTrue(SupportFunctions.DoubleCompare(matrix[1][4], 1.09));
        }



        #endregion CoVariance Tests
    }
}
