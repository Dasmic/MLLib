using System;
using System.Collections.Generic;
using Dasmic.MLLib.Math.Matrix;
using System.Threading.Tasks;


namespace Dasmic.MLLib.Algorithms.DiscriminantAnalysis
{
    public class ModelLinear:ModelBase
    {
        public double[] ClassProbabilities;
        public double[] ClassValues;
        public List<double[][]> ClassMeanMatrix;
        public double[][] PooledCovarianceMatrixInv ;

        public ModelLinear(double missingValue,
                                int indexTargetAttribute, 
                                int countAttributes) :
                                base(missingValue, 
                                    indexTargetAttribute, 
                                    countAttributes)
        {

        }

        public override double RunModelForSingleData(double[] data)
        {
            VerifyDataForRun(data);
            int maxIdx=0;
            double maxVal = _missingValue;
            double[] ldf = new double[ClassMeanMatrix.Count];
            
            //Modify data for double dimension array
            double[][] modData = new double[data.Length][];
            //for(int ii=0;ii<modData.Length;ii++)
            Parallel.For(0, modData.Length, ii =>
             {
                 modData[ii] = new double[] { data[ii] };
             });

            
            MatrixOperations mo = new MatrixOperations();
            modData = mo.Transpose(modData);
            //Apply the linear discriminant function
            for(int idx=0;idx < ClassMeanMatrix.Count;
                                idx++)
            {                
                double[][] tmp = mo.Multiply(ClassMeanMatrix[idx],
                                              PooledCovarianceMatrixInv);
                double[][]  tmp1 = mo.Multiply(tmp, modData);

                //double[][] tmp2 = mo.multiply(ClassMeanMatrix[idx],
                //                              PooledCovarianceMatrixInv);
                double[][] tmp2 = mo.Multiply(tmp, mo.Transpose(ClassMeanMatrix[idx]));

                ldf[idx] = tmp1[0][0] - (tmp2[0][0]/2.0) +
                    System.Math.Log(this.ClassProbabilities[idx]);

                if(maxVal==_missingValue)
                {
                    maxVal = ldf[idx];
                    maxIdx = idx;
                }
                else if(ldf[idx] > maxVal)
                {
                    maxVal = ldf[idx];
                    maxIdx = idx;
                }
            }

            return ClassValues[maxIdx];
        }

        //Serialization Routine
        public override void SaveModel(string filePath)
        {

        }

        //Deserialization Routine
        public override void LoadModel(string filePath)
        {

        }

    }
}
