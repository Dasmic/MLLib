using Dasmic.MLLib.UnitTest.Core;

namespace UnitTests.MLLib.Algorithms.SupportVectorMachine
{
    public class BaseTest : UnitTestBase
    {
        protected void initData_dataset_linear_subgd_jason_example()
        {
            _attributeHeaders = new string[] {
                                     "X1",
                                     "X2",
                                    "Y"};
            _indexTargetAttribute = 2;


            _trainingData = new double[3][];
            _trainingData[0] = new double[] {
                2.327868056,3.032830419,4.485465382,
                3.684815246,2.283558563,7.807521179,
                6.132998136,7.514829366,5.502385039,
                7.432932365};
            _trainingData[1] = new double[] {
                2.458016525,3.170770366,3.696728111,
                3.846846973,1.853215997,3.290132136,
                2.140563087,2.107056961,1.404002608,
                4.236232628 };
            _trainingData[2] = new double[] {
                -1,-1,-1,-1,-1,1,1,1,1,1
               };
        }

    }
}
