
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Collections.Generic;
using Dasmic.MLLib.Math.Statistics;

namespace UnitTests.MLLib.Math.Statistics
{
    [TestClass]
    public class InformationGainTest
    {
        double[] ta;

        public InformationGainTest()
        {
            setData();
        }

        /// <summary>
        /// R: 
        /// ta<-c(1,2,3,1,3,5,4,2,1,3,2,4,2,2,3,4,4)
        /// </summary>
        private void setData()
        {
            //15 elements
            ta = new double[] { 1, 2, 3, 1, 3, 5, 4, 2, 1,
                3, 2, 4, 2, 2, 3, 4, 4 };
        }


        /// <summary>
        /// R: 
        /// table(ta)
        /// </summary>
        [TestMethod]
        public void Statistics_Frequency()
        {
            Dictionary<double, long> freqs=
                InformationGain.Frequency(ta);

            Assert.AreEqual(freqs[2], 5);
            Assert.AreEqual(freqs[3], 4);
            Assert.AreEqual(freqs[4], 4);
        }

        [TestMethod]
        public void Statistics_EntropyShannon()
        {
            double value =
                InformationGain.EntropyShannon(ta);
            Assert.IsTrue( 2.17 <= value && value <= 2.19);   
        }


        [TestMethod]
        public void Statistics_EntropyShannon_ProvideFreq()
        {
            Dictionary<double, long> freqs =
                InformationGain.Frequency(ta);

            double value =
                InformationGain.EntropyShannon(freqs);

            Assert.IsTrue(2.17 <= value && value <= 2.19);
        }

    }
}
