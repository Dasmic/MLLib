using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Dasmic.MLLib.Algorithms.DecisionTree
{
    public class DecisionTreeNode
    {
        private Dictionary<double,DecisionTreeNode> 
            _children;
        private string _attributeName;
        private int _attributeIndex;
        private double _value;
        private DecisionTreeNode _parent;
        //private long _numberOfRows; //Used by CART to determine if node had ended

        public DecisionTreeNode(DecisionTreeNode parent)
        {
            Initialize();
            _value = Double.PositiveInfinity;
            _parent = parent;
        }

        private void Initialize()
        {
            _attributeIndex = -1;
            _attributeName = "";
        }


        //For CART stores value for each node
        //For ID3/C45 only used for final value
        //For ID3/C45 this value is not used as children are stored
        //in Dictionary indexed by value
        public double Value
        {
            get
            {
                return _value;
            }
            set
            {
                _value = value;
            }
        }


        /*public long NumberOfRows
        {
            get { return _numberOfRows; }
            set { _numberOfRows = value; }
        }*/

        //dtn is child node
        //Each child is stored with a value in the parent
        public void addChild(double value, DecisionTreeNode dtn) //Goto the child if value
        {
            if (_children == null)
                _children = new Dictionary<double, 
                    DecisionTreeNode>(); //Create new map is non exists
            
            _children.Add(value,dtn);
        }

        public List<DecisionTreeNode> Children
        {
            get
            {
                if (_children == null) return null;
                return _children.Values.ToList();
            }
        }



        public DecisionTreeNode getChildWithValue(double value)
        {
            if (_children == null) return null;
            if (!_children.ContainsKey(value)) return null;
            return _children[value];
        }

        public int AttributeIndex
        {
            get { 
                return _attributeIndex;
            }
        }

        public void setAttributeValues(int attributeIndex,
                            string attributeName)
        {
            _attributeIndex = attributeIndex;
            _attributeName = attributeName;
        }

  

        public string AttributeName
        {
            get
            {
                return _attributeName;
            }
        }



        public DecisionTreeNode Parent
        {
            get
            {
                return _parent;
            }
           
        }
    }
}
