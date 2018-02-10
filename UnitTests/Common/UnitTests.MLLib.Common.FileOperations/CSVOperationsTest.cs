using System;
using System.IO;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Dasmic.MLLib.Common.IO;
using Dasmic.MLLib.Common.IO.Windows;
using Dasmic.MLLib.Common.Exceptions;

namespace UnitTests_FileOperations
{
    [TestClass]
    public class CSVOperationsTest
    {
        double[][] _trainingData;
        string[] _attributeHeaders;

        private void InitData()
        {
            _attributeHeaders = new string[] { "Outlook",
                                    "Temperature",
                                    "Humidity",
                                    "Wind",
                                    "PlayTennis"};

            _trainingData = new double[5][];
            //0-Sunny,1-OverCast,2-Rain
            _trainingData[0] = new double[] { 0, 0, 1, 2, 2, 2, 1, 0, 0, 2, 0, 1, 1, 2 };
            //0-Hot, 1-Mild, 2-Cool
            _trainingData[1] = new double[] { 0, 0, 0, 1, 2, 2, 2, 1, 2, 1, 1, 1, 0, 1 };
            //0-Normal, 1-High
            _trainingData[2] = new double[] { 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1 };
            //0-Weak,1-Strong
            _trainingData[3] = new double[] { 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1 };
            //0-No,1-Yes
            _trainingData[4] = new double[] { 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0 };

            
        }

        public CSVOperationsTest()
        {
            InitData();
        }


        /// <summary>
        /// Load in R:
        ///  mydata = read.csv("C:\\Dasmic\\Development\\NET\\MachineLearning\\UnitTests\\FileOperationsLibTest\\bin\\Debug\\CSVFileWriteTest.csv")
        ///  output.tree<-ctree(PlayTennis~.Outlook+Temperature+Humidity+Wind,data=mydata)
        ///  plot(output.tree)
        /// </summary>
        [TestMethod]
        public void CSV_file_write()
        {
            IFileOperations csvOps = new CSVOperations();
            string path= AppDomain.CurrentDomain.SetupInformation.ApplicationBase;
            path = path + "\\CSVWriteTest.csv";
            //Delete file
            if (File.Exists(path))
                File.Delete(path);

            try
            { 
                csvOps.Write(_trainingData,
                    _attributeHeaders, path);
            }
            catch
            {
                Assert.Fail();
                return;
            }
            Assert.IsTrue(File.Exists(path));
        }

        [TestMethod]
        public void CSV_file_read()
        {
            IFileOperations csvOps = new CSVOperations();
            string path = AppDomain.CurrentDomain.SetupInformation.ApplicationBase;
            path = path + "\\CSVReadTest.csv";
            FileData fd;
            try
            {
                fd = csvOps.Read(path,-1);
            }
            catch
            {
                Assert.Fail();
                return;
            }
            Assert.AreEqual(fd.attributeHeaders.Length,3);
            Assert.AreEqual(fd.values[0].Length, 31);
        }

        /// <summary>
        /// One Row is:
        /// X,Y
        /// </summary>
        [TestMethod]
        [ExpectedException(typeof(InvalidDataSetFileException))]
        public void CSV_file_read_invalid_1_raises_exception()
        {
            IFileOperations csvOps = new CSVOperations();
            string path = AppDomain.CurrentDomain.SetupInformation.ApplicationBase;
            path = path + "\\CSVReadInvalidTest.csv";
            FileData fd;
            
            fd = csvOps.Read(path, -1);                                   
        }


        /// <summary>
        /// One row is:
        /// X,Y,
        /// </summary>
        [TestMethod]
        [ExpectedException(typeof(InvalidDataSetFileException))]
        public void CSV_file_read_invalid_2_raises_exception()
        {
            IFileOperations csvOps = new CSVOperations();
            string path = AppDomain.CurrentDomain.SetupInformation.ApplicationBase;
            path = path + "\\CSVReadInvalid2Test.csv";
            FileData fd;

            fd = csvOps.Read(path, -1);
        }
    }
}
