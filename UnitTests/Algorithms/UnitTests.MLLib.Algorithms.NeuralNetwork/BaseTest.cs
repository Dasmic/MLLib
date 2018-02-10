using Dasmic.MLLib.UnitTest.Core;

namespace UnitTests.MLLib.Algorithms.NeuralNetwork
{
    public class BaseTest : UnitTestBase
    {
        /// <summary>
        /// This dataset is not linearly separable
        /// </summary>
        protected void initData_dataset_naive_bayes_jason_example()
        {
            _attributeHeaders = new string[] {
                                     "Weather",
                                     "Car",
                                    "Class"};
            _indexTargetAttribute = 2;

            _trainingData = new double[3][];
            _trainingData[0] = new double[] { 1, 0, 1, 1, 1, 0, 0, 1, 1, 0 };
            _trainingData[1] = new double[] { 1, 0, 1, 1, 1, 0, 0, 1, 0, 0 };
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

        protected void initData_NN_dataset_linear_subgd_jason_example()
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

        /// <summary>
        //Initialize data from Jason's book
        /// R:
        /// X = list(c(1, 2, 4, 3, 5))
        /// Y = list(c(1, 3, 3, 2, 5))
        /// </summary>
        protected void Init_dataset_jason_linear_regression()
        {
            _attributeHeaders = new string[] {
                                     "X",
                                     "Y"};
            _indexTargetAttribute = 1;

            _trainingData = new double[2][];
            _trainingData[0] = new double[] { 1, 2, 4, 3, 5 };
            _trainingData[1] = new double[] { 1, 3, 3, 2, 5 };
        }

        /// <summary>
        /// Load the Pythagoras Data set
        /// </summary>
        protected void Init_dataset_pythagoras()
        {
            LoadFromDataSet(EnumDataSets.Pythagoras, -1);
        }

    }
}
