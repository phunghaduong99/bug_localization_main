import java.util.*;
import java.lang.reflect.*;
import org.aspectj.lang.annotation.*;


// JUST DONT ASK HOW THIS WORKS

abstract aspect ParentChildRelationship<Parent,Child> {

  interface ParentHasChildren<C>{
    List<C> getChildren();
    void addChild(C child);
    void removeChild(C child);
  }

  interface ChildHasParent<P>{
    P getParent();
    void setParent(P parent);
  }

  declare parents: Parent implements ParentHasChildren<Child>;
  declare parents: Child  implements ChildHasParent<Parent>;

  public List<E> ParentHasChildren<E>.children = new ArrayList<E>();
  public P ChildHasParent<P>.parent;

  public List<D> ParentHasChildren<D>.getChildren() {
    return Collections.unmodifiableList(children);  
  }

  public P ChildHasParent<P>.getParent() {
    return parent;
  }

  public void ChildHasParent<R>.setParent(R parent) {
    this.parent = parent;
    ((ParentHasChildren)parent).addChild(this);
  }

  public void ParentHasChildren<X>.addChild(X child) {
    if (((ChildHasParent)child).parent != null) {
      ((ParentHasChildren)((ChildHasParent)child).parent).removeChild(child);
    }
    children.add(child);
  }

  public void ParentHasChildren<Y>.removeChild(Y child) {
    if (children.remove(child)) {
      ((ChildHasParent)child).parent = null;
    }
  }

}

aspect GenericAspectU extends ParentChildRelationship<Top,Bottom> { 

  public static void main(String []argv) {

    // Check the state of top
    Top t = new Top();
    check(t instanceof ParentHasChildren,"Top should implement ParentHasChildren");
    Type[] intfs = Top.class.getGenericInterfaces();
    check(intfs[0] instanceof ParameterizedType,
          "Expected Top to have parameterized interface but found "+intfs[0]);
    ParameterizedType pt = (ParameterizedType) intfs[0];
    Type[] tArgs = pt.getActualTypeArguments();
    check(tArgs[0]==Bottom.class,
          "Expecting Bottom parameter but found " + tArgs[0]);


    // Check the state of top
    Bottom b = new Bottom();
    check(b instanceof ChildHasParent,"Bottom should implement ChildHasParent");
    intfs = Bottom.class.getGenericInterfaces();
    check(intfs[0] instanceof ParameterizedType,
          "Expected Bottom to have parameterized interface but found "+intfs[0]);
    pt = (ParameterizedType) intfs[0];
    tArgs = pt.getActualTypeArguments();
    check(tArgs[0]==Top.class,
          "Expecting Top parameter but found " + tArgs[0]);




    // Field fiddling
    b.parent = t;
    List<Bottom> kids = new ArrayList<Bottom>();
    kids.add(b);
    t.children = kids;


    // start using the methods
    List<Bottom> kids2 = t.getChildren();
    check(kids2.size()==1,
      "Expected one child of the Top but found "+kids2.size());
    check(kids2.get(0).equals(b),
      "Expected one child of the Top which was what we put in there!"+kids2.get(0));

    // and the parent methods
    Top retrievedParent = b.getParent();
    check(retrievedParent==t,
      "parent check 1 failed "+
      "retrieved="+retrievedParent+"  expected="+t);


    Top top2 = new Top();
    b.setParent(top2);
    Top retrievedParent2 = b.getParent();
    check(retrievedParent2==top2,
      "parent check 2 failed "+
      "retrieved="+retrievedParent2+"  expected="+top2);
    
    Top top3 = new Top();
    Bottom bot2 = new Bottom();
    top3.addChild(bot2);
    Bottom aBottom = top3.getChildren().get(0);
    check(aBottom==bot2,"Incorrect child? expected="+bot2+" found="+aBottom);
    top3.removeChild(bot2);
    int size=top3.getChildren().size();
    check(size==0,"Should be no children but there were "+size);


  }

  public static void check(boolean b,String msg) {
    if (!b) throw new RuntimeException(msg);
  }
}

class Top {}
class Bottom {}

//////////////////////////////////////////////////////////////////

/* End game for test Z, as bits work they are promoted up into the 
   testcase above :)

   TestN promoted the declare parents statements up
   TestO promoted the fields up - a parent knows its children, a 
         child knows its parents - but then used them incorrectly
   TestP uses the fields correctly
   TestQ ... tests some stumbling blocks I encountered before R...
   TestR promoted getChildren() method
   TestS promoted getParent() and setParent()
   TestT ... tests some stumbling blocks I encountered before U...
   TestU promoted addChild and removeChild

public abstract aspect ParentChildRelationship<Parent,Child> {

    @SuppressAjWarnings
    public pointcut addingChild(Parent p, Child c) :
      execution(* Parent.addChild(Child)) && this(p) && args(c);
      
    @SuppressAjWarnings
    public pointcut removingChild(Parent p, Child c) :
      execution(* Parent.removeChild(Child)) && this(p) && args(c);
}
*/

