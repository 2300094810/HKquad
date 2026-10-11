// Isolated CLI for the GPLv3 libQEx reference extractor. See THIRD_PARTY.md.
#include <qex.h>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  try {
    if (argc != 3) throw std::runtime_error("usage: hktex2quad_qex input.txt output.obj");
    std::ifstream in(argv[1]);
    std::string magic;
    unsigned int nv = 0, nf = 0;
    in >> magic >> nv >> nf;
    if (magic != "HKTEX2QUAD_QEX_V1" || !nv || !nf) throw std::runtime_error("invalid QEx input");
    std::vector<qex_Point3> vertices(nv);
    std::vector<qex_Tri> triangles(nf);
    std::vector<qex_UVTri> uv(nf);
    for (auto& v : vertices) for (double& x : v.x) in >> x;
    for (auto& f : triangles) for (auto& i : f.indices) {
      in >> i;
      if (i >= nv) throw std::runtime_error("invalid vertex index");
    }
    for (auto& f : uv) for (auto& p : f.uvs) for (double& x : p.x) in >> x;
    if (!in) throw std::runtime_error("truncated QEx input");
    // Use the OpenMesh API: the legacy raw C wrapper assumes face-circulator
    // order equals input order, and leaves uninitialized slots for non-quads.
    QEx::TriMesh mesh;
    for (const auto& v : vertices) mesh.add_vertex(QEx::TriMesh::Point(v.x[0], v.x[1], v.x[2]));
    std::vector<QEx::TriMesh::FaceHandle> handles;
    for (const auto& f : triangles) {
      auto face = mesh.add_face(mesh.vertex_handle(f.indices[0]), mesh.vertex_handle(f.indices[1]), mesh.vertex_handle(f.indices[2]));
      if (!face.is_valid()) throw std::runtime_error("QEx requires an oriented manifold input");
      handles.push_back(face);
    }
    std::vector<OpenMesh::Vec2d> halfedge_uv(mesh.n_halfedges());
    for (unsigned int fi = 0; fi < nf; ++fi) {
      for (auto he = mesh.fh_begin(handles[fi]); he != mesh.fh_end(handles[fi]); ++he) {
        const auto vertex = static_cast<unsigned int>(mesh.to_vertex_handle(*he).idx());
        for (unsigned int corner = 0; corner < 3; ++corner) {
          if (triangles[fi].indices[corner] == vertex)
            halfedge_uv[he->idx()] = OpenMesh::Vec2d(uv[fi].uvs[corner].x[0], uv[fi].uvs[corner].x[1]);
        }
      }
    }
    QEx::QuadMesh out;
    QEx::extractQuadMeshOM(&mesh, &halfedge_uv, nullptr, &out);
    // QEx can close an open surface's exterior loop with a spurious n-gon.
    // Remove ONLY a cap whose every edge lies on the original open boundary
    // and whose opposite faces are quads. Never drop interior failed cells.
    std::vector<std::pair<QEx::TriMesh::Point, QEx::TriMesh::Point>> boundary;
    for (auto he = mesh.halfedges_begin(); he != mesh.halfedges_end(); ++he)
      if (mesh.is_boundary(*he)) boundary.push_back({mesh.point(mesh.from_vertex_handle(*he)), mesh.point(mesh.to_vertex_handle(*he))});
    double extent = 0;
    for (const auto& v : vertices) for (double x : v.x) extent = std::max(extent, std::abs(x));
    const double tolerance = std::max(1., extent) * 1e-8;
    auto on_boundary = [&](const QEx::QuadMesh::Point& point, const QEx::QuadMesh::Point& direction) {
      for (const auto& segment : boundary) {
        auto d = segment.second - segment.first;
        const double t = std::max(0., std::min(1., OpenMesh::dot(point - segment.first, d) / std::max(1e-30, d.sqrnorm())));
        if ((point - (segment.first + d * t)).norm() <= tolerance && OpenMesh::dot(direction, d) > 0) return true;
      }
      return false;
    };
    std::vector<QEx::QuadMesh::FaceHandle> caps;
    for (auto face = out.faces_begin(); face != out.faces_end(); ++face) {
      if (boundary.empty()) continue;
      bool cap = true;
      for (auto he = out.fh_begin(*face); he != out.fh_end(*face); ++he) {
        auto a = out.point(out.from_vertex_handle(*he));
        auto b = out.point(out.to_vertex_handle(*he));
        auto other = out.face_handle(out.opposite_halfedge_handle(*he));
        // Original boundary halfedges follow the exterior winding, opposite
        // to legitimate surface faces. This also detects a four-sided cap.
        cap = cap && other.is_valid() && out.valence(other) == 4 && on_boundary(a, b-a) && on_boundary((a + b) * .5, b-a);
      }
      if (cap) caps.push_back(*face);
    }
    if (!caps.empty()) {
      out.request_face_status(); out.request_edge_status(); out.request_halfedge_status(); out.request_vertex_status();
      for (auto face : caps) out.delete_face(face, false);
      out.garbage_collection();
      std::cout << "removed_exterior_boundary_caps=" << caps.size() << '\n';
    }
    unsigned int nonquads = 0;
    for (auto face = out.faces_begin(); face != out.faces_end(); ++face) {
      if (out.valence(*face) != 4) {
        std::cerr << "Non-quad face " << face->idx() << " valence=" << out.valence(*face) << '\n';
        ++nonquads;
      }
    }
    if (nonquads) throw std::runtime_error("QEx returned non-quad faces; output rejected");
    std::ofstream obj(argv[2]);
    obj << std::setprecision(17) << "# libQEx integer-grid extraction\n";
    for (auto v = out.vertices_begin(); v != out.vertices_end(); ++v) {
      const auto& p = out.point(*v);
      obj << "v " << p[0] << ' ' << p[1] << ' ' << p[2] << '\n';
    }
    for (auto face = out.faces_begin(); face != out.faces_end(); ++face) {
      obj << "f";
      for (auto v = out.fv_begin(*face); v != out.fv_end(*face); ++v) obj << ' ' << v->idx() + 1;
      obj << '\n';
    }
    std::cout << "vertices=" << out.n_vertices() << " quads=" << out.n_faces() << '\n';
    if (!obj) throw std::runtime_error("cannot write QEx output");
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n'; return 1;
  }
}
