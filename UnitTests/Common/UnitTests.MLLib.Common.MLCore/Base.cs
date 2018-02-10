using System.Collections.Generic;
using Dasmic.MLLib.UnitTest.Core;

namespace UnitTests.MLLib.Common.MLCore
{
    public class Base:UnitTestBase
    {
        protected double[][] _classMatrix1;
        protected double[][] _classMatrix2;
        protected List<double[][]> _classInputMatrix;


        /// <summary>
        /// Using example from site
        /// 
        /// http://people.revoledu.com/kardi/tutorial/LDA/Numerical%20Example.html
        /// </summary>
        protected void initData_dataset_7_row_2_class_example()
        {
            _attributeHeaders = new string[] {
                                     "Attr 1",
                                    "Attr 2",
                                    "Value"};
            _indexTargetAttribute = 2;

            _trainingData = new double[3][];
            _trainingData[0] = new double[] { 2.95, 2.53, 3.57, 3.16, 2.58, 2.16, 3.27 };
            _trainingData[1] = new double[] { 6.63, 7.79, 5.65, 5.47, 4.46, 6.22, 3.52 };
            _trainingData[2] = new double[] { 1, 1, 1, 1, 2, 2, 2, };

            _classMatrix1 = new double[2][];
            _classMatrix2 = new double[2][];
            _classMatrix1[0] = new double[] { 2.95, 2.53, 3.57, 3.16 };
            _classMatrix1[1] = new double[] { 6.63, 7.79, 5.65, 5.47 };

            _classMatrix2[0] = new double[] { 2.58, 2.16, 3.27 };
            _classMatrix2[1] = new double[] { 4.46, 6.22, 3.52 };

            _classInputMatrix =
                new List<double[][]>();

            _classInputMatrix.Add(_classMatrix1);
            _classInputMatrix.Add(_classMatrix2);
        }

        protected void initData_classdata_4_rows_2_class()
        {
            _classInputMatrix =
                new List<double[][]>();
            
            _attributeHeaders = new string[] {
                                     "Attr 1",
                                    "Attr 2",
                                    "Value"};

            _trainingData = new double[3][];
            _trainingData[0] = new double[] { 100, 60, 30, 50 };
            _trainingData[1] = new double[] { 200, 80, 92, 58 };
            _trainingData[2] = new double[] { 1, 0, 1, 0 };
            _indexTargetAttribute = 2;

            _classMatrix1 = new double[2][];
            _classMatrix1[0] = new double[] { 100, 60 };
            _classMatrix1[1] = new double[] { 200, 80 };

            _classMatrix2 = new double[2][];
            _classMatrix2[0] = new double[] { 30, 50 };
            _classMatrix2[1] = new double[] { 92, 58 };

            _classInputMatrix.Add(_classMatrix1);
            _classInputMatrix.Add(_classMatrix2);

            _indexTargetAttribute = 2;
        }
    }
}
