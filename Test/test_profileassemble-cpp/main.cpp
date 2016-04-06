#include <dolfin.h>
#include "test_profileassemble.h"

using namespace dolfin;


class Termf : public Expression
{
    void eval(Array<double>& values, const Array<double>& x) const
    {
        values[0] = 1.0;
    }
};


class Termk : public Expression
{
    public:
        Termk() : nb(1.0) {}

        void update()
        {
            nb++;
        }

    private:
        double nb;

        void eval(Array<double>& values, const Array<double>& x) const
        {
            values[0] = nb;
        }
};

int main()
{
    #ifdef HAS_PETSC
    UnitSquareMesh mesh(100,100);
    test_profileassemble::FunctionSpace V(mesh);

    /*Termf f;
    test_profileassemble::LinearForm L(V);
    L.f = f;
    Vector x;
    assemble(x, L);
    Vector y(x);*/

    Termk k;
    test_profileassemble::BilinearForm a(V,V);
    a.k = k;
    std::shared_ptr<GenericMatrix> A(new Matrix);

    for (int ii=0; ii<200; ii++)
    {
        assemble(*A, a);
        k.update();
    }

    #else
    cout << "Need PETSc." << endl;
    #endif
}
