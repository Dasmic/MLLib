using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Dasmic.MLLib.Algorithms.DeepLearning.Support;

namespace Dasmic.MLLib.Algorithms.DeepLearning
{
    public abstract class ModelConvNetBase : MLLib.Common.MLCore.ModelBase
    {
        private LayerBase[] _layers { get; set; }

        public enum EnumMode
        {
            Regression,
            Classification
        }

        public EnumMode Mode;
        public double[][] WeightsHidden { get; set; }
        public double[] TargetValues;

        public ModelConvNetBase(double missingValue,
                        int indexTargetAttribute,
                        int countAttributes) :
                                base(missingValue, 
                                    indexTargetAttribute, countAttributes)
        {
            Mode = EnumMode.Regression;
        }

        /// <summary>
        /// Pass dummy values to base, make sure to call SetValues 
        /// to pass values
        /// </summary>
        public ModelConvNetBase():base(0,0,0)
        {

        }

       
        //public override abstract double RunModelForSingleData(double[] data);

        /// <summary>
        /// Will return single value which is the maximum value
        /// among all output units
        /// </summary>
        /// <param name="data"></param>
        /// <returns></returns>
        public override double RunModelForSingleData(double[] data)
        {
            double[] computedValue = GetOutputValues(data);

            //Find max Idx in computedValue
            int maxIdx = computedValue.Select((item, indx) =>
                new { Item = item, Index = indx }).
                OrderByDescending(x => x.Item).Select(x => x.Index).First();

            if (Mode == EnumMode.Classification)
            {
                return TargetValues[maxIdx];
            }
            else
                return computedValue[maxIdx];
        }

        /// <summary>
        /// Returns an array with output values of 
        /// all computed units
        /// </summary>
        /// <param name="data"></param>
        /// <returns></returns>
        public double[] GetOutputValues(double[] data)
        {
            VerifyDataForRun(data);
           
        }
    }
}
