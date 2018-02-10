using Dasmic.MLLib.UnitTest.Core;

namespace UnitTests.MLLib.Math.Statistics
{
    public class BaseTest: UnitTestBase
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
    }
}
