using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Dasmic.MLLib.Algorithms.DiscriminantAnalysis;
using Dasmic.MLLib.UnitTest.Core;

namespace UnitTests_DiscriminantAnalysis
{
    public class BaseTest:UnitTestBase
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


        /// <summary>
        /// 
        /// </summary>
        protected void initData_dataset_40_row_1_jason_example()
        {
            
            _attributeHeaders = new string[] {
                                     "Attr 1",
                                    "Value"};
            _indexTargetAttribute = 1;

            
            _trainingData = new double[2][];
            _trainingData[0] = new double[] {
                4.667797637,
                5.509198779,
                4.702791608,
                5.956706641,
                5.738622413,
                5.027283325,
                4.805434058,
                4.425689143,
                5.009368635,5.116718815,6.370917709,2.895041947,4.666842365,
                5.602154638,4.902797978,5.032652964,4.083972925,4.875524106,
                4.732801047,5.385993407,20.74393514,21.41752855,20.57924186,
                20.7386947,19.44605384,18.36360265,19.90363232,19.10870851,
                18.18787593,19.71767611,19.09629027,20.52741312,
                20.63205608,19.86218119,21.34670569,20.333906,21.02714855,
                18.27536089,21.77371156,20.65953546};
            _trainingData[1] = new double[] {
                     0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,
                        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};

        }

        //This assume assignment to local class values is done
        protected BuildBase getBuildAndAssignValues()
        {
            BuildBase lda =
                   new BuildLinear();
            setPrivateVariablesInBuildObject(lda);           
            return lda;
        }

    }
}
