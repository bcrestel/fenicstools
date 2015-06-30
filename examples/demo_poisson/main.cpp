// Copyright (C) 2006-2011 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2006-02-07
// Last changed: 2013-03-11
//
// This demo program solves Poisson's equation
//
//     - div grad u(x, y) = f(x, y)
//
// on the unit square with source f given by
//
//     f(x, y) = 10*exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.02)
//
// and boundary conditions given by
//
//     u(x, y) = 0        for x = 0 or x = 1
// du/dn(x, y) = sin(5*x) for y = 0 or y = 1

#include <iostream>

#include <dolfin.h>
#include "Poisson.h"

using namespace dolfin;

// Source term (right-hand side)
class Source : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    double dx = x[0] - 0.5;
    double dy = x[1] - 0.5;
    values[0] = 10*exp(-(dx*dx + dy*dy) / 0.02);
  }
};

// Normal derivative (Neumann boundary condition)
class dUdN : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = sin(5*x[0]);
  }
};

// Sub domain for Dirichlet boundary condition
class DirichletBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS;
  }
};

int main()
{
  MPI_Comm mycomm = MPI_COMM_WORLD;
  int myrank = MPI::rank(mycomm);

  //PETScOptions::set("log_summary", "profile.txt");

  Timer timer1("Partition the mesh and define function space"); 
  // Create mesh and function space
  timer1.start();
  UnitSquareMesh mesh(1000, 1000);
  Poisson::FunctionSpace V(mesh);
  timer1.stop();

  Timer timer2("Define BCs and assemble problem");
  // Define boundary condition
  timer2.start();
  Constant u0(0.0);
  DirichletBoundary boundary;
  DirichletBC bc(V, u0, boundary);

  // Define variational forms
  Poisson::BilinearForm a(V, V);
  Poisson::LinearForm L(V);

  Source f;
  dUdN g;
  L.f = f;
  L.g = g;

  // Compute solution
  Function u(V);
  timer2.stop();
  Timer timer3("Solve problem");
  timer3.start();
  solve(a == L, u, bc);
  timer3.stop();

  if (myrank == 0) {
  std::cout << "Time to solve problem: " << timing("Solve problem") << std::endl;}

  /*/ Save solution in VTK format
  File file("poisson.pvd");
  file << u;

  // Plot solution
  plot(u);
  interactive();*/

  return 0;
}

/* Profiling of that code (based on "time to solve problem" only:

nb proc =   1       2        3        4
        37.7568  17.3586  16.0754  13.5388
        37.7182  18.2116  16.375   13.7601
        37.8255  17.4927  16.2503  13.5594
