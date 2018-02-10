using System;
using Dasmic.MLLib.Common.MLCore;
using Dasmic.MLLib.Common.Exceptions;


namespace Dasmic.MLLib.Algorithms.DecisionTree
{
    public class ModelCART : ModelBase
    {
        public ModelCART(double missingValue,
                                int indexTargetAttribute,
                                int countAttributes) :
                                base(missingValue,
                                    indexTargetAttribute,
                                    countAttributes)
        { }

        /// <summary>
        /// Return value of a target attribute. 
        /// Main assumption is that data is in same indexes as training data
        /// </summary>
        /// <param name="data"></param>
        /// <param name="indexTargetAttribute"></param>
        /// <returns></returns>
        public override
            double RunModelForSingleData(double[] data)
        {
            try
            {
                double value = double.PositiveInfinity;
                if (_root == null) return value; //Model not computed
                DecisionTreeNode dtn = _root;
                
                //Deal with missing values
                while (dtn.AttributeIndex != _origTargetAttributeIndex) //dtn.AttributeIndex < 0 denoted invalid node
                {
                    if(data[dtn.AttributeIndex] < dtn.Value)
                        //Children are stored with Index
                        dtn = dtn.getChildWithValue(0);
                    else
                        dtn = dtn.getChildWithValue(1);

                    if (dtn == null) //Is null likely due to missing value, ignore it in this case
                    {
                        return Constants.MISSING_VALUE;
                    }
                }
                value = dtn.Value;
                return value;
            }
            catch (Exception ex)
            {
                throw new ModelRunException(ex.Message, ex);
            }
        }

        /// <summary>
        /// Return number of correct estimation in full data set 
        /// Main assumption is that data is in same indexes as training data
        /// 
        /// Use this routine, instead of the routine for each individual row
        /// as this is more efficient
        /// </summary>
        /// <param name="data">Data should also have the target attribute, and same feature arrangement as training data</param>
        /// <returns></returns>
        public
            long getNumberOfCorrectPredictions(double[][] data)
        {
            long correctCount=0;
            if (_root != null) //Model not computed
            {
                try
                {
                    for (int rowIdx = 0; rowIdx < data[0].Length; rowIdx++)
                    {                        
                        DecisionTreeNode dtn = _root;

                        //Deal with missing values
                        while (dtn.AttributeIndex != _origTargetAttributeIndex) //dtn.AttributeIndex < 0 denoted invalid node
                        {
                            if (data[dtn.AttributeIndex][rowIdx] < dtn.Value)
                                //Children are stored with Index
                                dtn = dtn.getChildWithValue(0);
                            else
                                dtn = dtn.getChildWithValue(1);

                            if (dtn == null) //Is null likely due to missing value, ignore it in this case
                            {
                                break;
                            }
                        }
                        if (dtn.Value == data[_origTargetAttributeIndex][rowIdx])
                            correctCount++;

                    }
                }
                catch (Exception ex)
                {
                    throw new ModelRunException(ex.Message, ex);
                }
            }
            return correctCount;
        }
    }
}
