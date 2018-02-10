using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Dasmic.MLLib.Common.Exceptions
{
    public class AttributesCountMismatchException : Exception
    {
        public override string Message
        {
            get
            {
                return Resources.strings_messages.exception_attribute_mismatch;
            }
        }
    }
}
