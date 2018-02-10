using Dasmic.MLLib.UnitTest.Core;

namespace UnitTests.MLLib.Algorithms.InstanceBased
{
    public class BaseTest : UnitTestBase
    {
        /// <summary>
        /// Example from:
        /// https://math.la.asu.edu/~gardner/QR.pdf
        /// R:
        /// A = matrix(c(1,0,1,0,1,2,1,2,0),nrow=3,ncol=3)
        /// </summary>
        protected void InitData_dataset_3_rows_symmetric_hessenberg()
        {
            _trainingData = new double[3][];
            _trainingData[0] = new double[] { 1, 0, 1 };
            _trainingData[1] = new double[] { 0, 1, 2 };
            _trainingData[2] = new double[] { 1, 2, 0 };
        }


        /// <summary>        
        /// R:
        /// A = matrix(c(1,0,0,1),nrow=2,ncol=2)
        /// </summary>
        protected void InitData_dataset_2_rows_2_category()
        {
            _attributeHeaders = new string[] {
                                     "X",
                                     "Y"};
            _indexTargetAttribute = 1;
            _trainingData = new double[2][];
            _trainingData[0] = new double[] { 1, 0 };
            _trainingData[1] = new double[] { 0, 1 };            
        }

        /// <summary>        
        /// R:
        /// A = matrix(c(1,1,1,1),nrow=2,ncol=2)
        /// </summary>
        protected void InitData_dataset_2_rows_1_category()
        {
            _attributeHeaders = new string[] {
                                     "X",
                                     "Y"};
            _indexTargetAttribute = 1;
            _trainingData = new double[2][];
            _trainingData[0] = new double[] { 1, 1 };
            _trainingData[1] = new double[] { 1, 1 };
        }


        /// <summary>        
        /// R:
        /// library(datasets)
        /// data(iris)
        /// write.csv(iris,"C:\\Users\\Chaitanya Belwal\\Documents\\iris.csv")
        /// </summary>
        protected void InitData_dataset_iris()
        {
            LoadFromDataSet(EnumDataSets.Iris, -1);
        }
    }
}
