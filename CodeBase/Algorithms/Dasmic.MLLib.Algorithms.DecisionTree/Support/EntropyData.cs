using System;
using System.Collections.Generic;


namespace Dasmic.MLLib.Algorithms.DecisionTree
{
    public class SplittedAttributeData
    {
        public Dictionary<double, long> Freqs;
        public int AttributeIndex;
        public double SplittingCriteriaValue;
    }
}
