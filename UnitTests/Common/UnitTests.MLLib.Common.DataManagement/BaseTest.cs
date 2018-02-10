using Dasmic.MLLib.UnitTest.Core;

namespace UnitTests.MLLib.Common.DataManagement
{
    public class BaseTest:UnitTestBase
    {
        protected string [][] _dataString;
        /// <summary>
        /// R:
        /// A = matrix(c(1,2,3,4,5,6,7,8,9),nrow=3,ncol=3)
        /// </summary>
        protected void InitData_dataset_3_rows_non_symmetric()
        {
            _attributeHeaders = new string[] {
                                     "X",
                                     "Y",
                                        "Z"};
            _indexTargetAttribute = 2;
            _trainingData = new double[3][];
            _trainingData[0] = new double[] { 1, 2, 3 };
            _trainingData[1] = new double[] { 4, 5, 6 };
            _trainingData[2] = new double[] { 7, 8, 9 };
        }

        protected void InitData_dataset_3_rows_string()
        {
            _attributeHeaders = new string[] {
                                     "X","Y","Z"};
            _indexTargetAttribute = 2;
            _dataString = new string[3][];
            _dataString[0] = new string[] { "a", "b", "c" };
            _dataString[1] = new string[] { "d", "e", "f" };
            _dataString[2] = new string[] { "g", "h", "i" };
        }
    }
}
