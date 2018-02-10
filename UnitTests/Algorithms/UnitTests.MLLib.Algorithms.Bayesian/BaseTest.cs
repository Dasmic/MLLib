using Dasmic.MLLib.Common.MLCore;
using Dasmic.MLLib.UnitTest.Core;


namespace UnitTests.MLLib.Algorithms.Bayesian
{
    public class BaseTest : UnitTestBase
    {
        //This assume assignment to local class values is done
        protected BuildBase getBuildAndAssignValues()
        {          
            return null;
        }

        protected void initData_dataset_naive_bayes_jason_example()
        {
            _attributeHeaders = new string[] {
                                     "Weather",
                                     "Car",
                                    "Class"};
            _indexTargetAttribute = 2;


            _trainingData = new double[3][];
            _trainingData[0] = new double[] {1,0,1,1,1,0,0,1,1,0};
            _trainingData[1] = new double[] {1,0,1,1,1,0,0,1,0,0};
            _trainingData[2] = new double[] { 1, 1, 1, 1, 1, 0, 0, 0, 0, 0 };

        }


        protected void initData_dataset_gaussian_naive_bayes_jason_example()
        {
            _attributeHeaders = new string[] {
                                     "X1",
                                     "X2",
                                    "Y"};
            _indexTargetAttribute = 2;


            _trainingData = new double[3][];
            _trainingData[0] = new double[] {
                3.393533211,3.110073483,1.343808831,
                3.582294042,2.280362439,7.423436942,
                5.745051997,9.172168622,7.792783481,
                7.939820817};
            _trainingData[1] = new double[] {
                2.331273381,1.781539638,3.368360954,
                4.67917911,2.866990263,4.696522875,
                3.533989803,2.511101045,3.424088941,
                0.791637231};
            _trainingData[2] = new double[] {
                0,0,0,0,0,1,1,1,1,1};
        }
    }
}
