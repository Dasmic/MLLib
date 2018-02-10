using Dasmic.MLLib.UnitTest.Core;

namespace UnitTests.MLLib.Algorithms.DimensionReduction
{
    public class BaseTest : UnitTestBase
    {
        // <summary>
        /// Dataset taken from:
        /// http://www.cs.otago.ac.nz/cosc453/student_tutorials/principal_components.pdf
        /// </summary>
        protected void InitData_dataset_pca_example()
        {
            _attributeHeaders = new string[] {
                                     "X",
                                     "Y"};
            _indexTargetAttribute = 1;

            _trainingData = new double[2][];
            _trainingData[0] = new double[] { 2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2, 1, 1.5, 1.1 };
            _trainingData[1] = new double[] { 2.4, 0.7, 2.9, 2.2, 3.0, 2.7, 1.6, 1.1, 1.6, 0.9 };

        }

        /// <summary>
        /// Example from:
        /// https://math.la.asu.edu/~gardner/QR.pdf
        /// R:
        /// A = matrix(c(1,0,1,0,1,2,1,2,0),nrow=3,ncol=3)
        /// </summary>
        protected void InitData_dataset_3_rows_symmetric_hessenberg()
        {
            _attributeHeaders = new string[] {
                                     "X",
                                     "Y","Z"};
            _indexTargetAttribute = 2;
            _trainingData = new double[3][];
            _trainingData[0] = new double[] { 1, 0, 1 };
            _trainingData[1] = new double[] { 0, 1, 2 };
            _trainingData[2] = new double[] { 1, 2, 0 };
        }

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

        /// <summary>
        /// Example from:
        /// http://web.csulb.edu/~tgao/math423/s93.pdf
        /// R:
        /// A = matrix(c(4,1,-2,2,1,2,0,1,-2,0,3,-2,2,1,-2,-1),nrow=4,ncol=4)
        /// </summary>
        protected void InitData_dataset_4_rows_symmetric_hessenberg()
        {
            _attributeHeaders = new string[] {
                                     "X",
                                     "Y","Z","A"};
            _indexTargetAttribute = 3;
            _trainingData = new double[4][];
            _trainingData[0] = new double[] { 4, 1, -2, 2 };
            _trainingData[1] = new double[] { 1, 2, 0, 1 };
            _trainingData[2] = new double[] { -2, 0, 3, -2 };
            _trainingData[3] = new double[] { 2, 1, -2, -1 };
        }
    }
}
