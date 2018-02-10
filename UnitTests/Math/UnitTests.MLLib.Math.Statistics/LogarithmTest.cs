using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Collections.Generic;
using Dasmic.MLLib.Math.Statistics;

namespace UnitTests.MLLib.Math.Statistics
{
    [TestClass]
    public class LogarithmTest
    {
        double[] ta;

        public LogarithmTest()
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
        public void Statistics_Log2()
        {
            Assert.AreEqual(Logarithm.Log2(2), 1);
            Assert.AreEqual(Logarithm.Log2(1), 0);
        }

        

    }
}
