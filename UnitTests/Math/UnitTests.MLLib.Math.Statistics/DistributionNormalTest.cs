using Dasmic.Portable.Core;
using Dasmic.MLLib.Math.Statistics;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace UnitTests.MLLib.Math.Statistics
{
    [TestClass]
    public class DistributionNormalTest
    {
        /// <summary>
        /// 
        /// </summary>
        [TestMethod]
        public void Statistics_Cdf_With_Seriesmax()
        {
            DistributionNormal snd =
                new DistributionNormal();
            double value = snd.CumulativeDistributionFunction(.5, 100);

            Assert.IsTrue(.691 <= value && value <= .692);
        }

        /// <summary>
        /// 
        /// </summary>
        [TestMethod]
        public void Statistics_Cdf_General()
        {
            DistributionNormal snd =
                new DistributionNormal();
            double value = snd.CumulativeDistributionFunction(.5);

            Assert.IsTrue(.691 <= value && value <= .692);
        }

        /// <summary>
        /// 
        /// </summary>
        [TestMethod]
        public void Statistics_Cdf_Zero()
        {
            DistributionNormal snd =
                new DistributionNormal();
            double value = snd.CumulativeDistributionFunction(0);

            Assert.IsTrue(.5 <= value && value <= .5);
        }

        /// <summary>
        /// 
        /// </summary>
        [TestMethod]
        public void Statistics_Cdf_Positive()
        {
            DistributionNormal snd =
                new DistributionNormal();
            double value = snd.CumulativeDistributionFunction(1);

            Assert.IsTrue(.840 <= value && value <= .842);
        }

        /// <summary>
        /// 
        /// </summary>
        [TestMethod]
        public void Statistics_Cdf_Negative()
        {
            DistributionNormal snd =
                new DistributionNormal();
            double value = snd.CumulativeDistributionFunction(0.0 - 1.0);

            Assert.IsTrue(.157 <= value && value <= .159);
        }

        /// <summary>
        /// Values compares with ZTable
        /// </summary>
        [TestMethod]
        public void Statistics_Zvalue_1sd()
        {
            DistributionNormal snd =
                new DistributionNormal();
            double value = snd.ZValue(34.13);

            Assert.IsTrue(0.99 <= value && value <= 1.1);
        }

        /// <summary>
        /// Values compares with ZTable
        /// </summary>
        [TestMethod]
        public void Statistics_zvalue_2sd()
        {
            DistributionNormal snd =
                new DistributionNormal();
            double value = snd.ZValue(47.72,.0001);

            Assert.IsTrue(1.98 <= value && value <= 2.1);
        }

        /// <summary>
        /// Values compares with ZTable
        /// </summary>
        [TestMethod]
        public void Statistics_Zvalue_Confidence_Internval()
        {
            DistributionNormal snd =
                new DistributionNormal();
            double value = snd.ZValue_ConfidenceInterval(25, .0001);

            Assert.IsTrue(.31 <= value && value <= .32);
        }

        [TestMethod]
        public void Statistics_Pdf_Example_1()
        {
            DistributionNormal snd =
                new DistributionNormal();
            double value = snd.ProbabilityDensityFunction(0, 0, 1);

            Assert.IsTrue(SupportFunctions.DoubleCompare(value,.398));
        }

        [TestMethod]
        public void Statistics_Pdf_Example_2()
        {
            DistributionNormal snd =
                new DistributionNormal();
            double value = snd.ProbabilityDensityFunction(-4, 0, 1);

            Assert.IsTrue(SupportFunctions.DoubleCompare(value, .0001));
        }

        [TestMethod]
        public void Statistics_Pdf_Example_3()
        {
            DistributionNormal snd =
                new DistributionNormal();
            double value = snd.ProbabilityDensityFunction(4, 0, 1);

            Assert.IsTrue(SupportFunctions.DoubleCompare(value, .0001));
        }

        /// <summary>
        /// Values compares with ZTable
        /// </summary>
        /*[TestMethod]
        public void TestGauss_1SD()
        {
            DistributionStandardNormal snd =
                new DistributionStandardNormal();
            double value = snd.Gauss(2.0);

             Assert.IsTrue(1.98 <= value && value <= 2.1);
        }*/

    }



}
