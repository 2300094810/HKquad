#include <igl/comb_cross_field.h>
#include <igl/comb_frame_field.h>
#include <igl/compute_frame_field_bisectors.h>
#include <igl/cross_field_mismatch.h>
#include <igl/cut_mesh_from_singularities.h>
#include <igl/find_cross_field_singularities.h>
#include <igl/copyleft/comiso/miq.h>

#include <Eigen/Core>

#include <cmath>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

struct Options {
  std::string input;
  std::string output;
  double gradient_scale = 1.0;
  double stiffness = 5.0;
  unsigned int iterations = 5;
  unsigned int local_iterations = 5;
  bool direct_round = false;
  bool round_singularities = true;
};

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error(message);
}

double parse_double(const char *value, const std::string &name) {
  char *end = nullptr;
  const double parsed = std::strtod(value, &end);
  if (end == value || *end != '\0' || !std::isfinite(parsed)) {
    fail("invalid " + name + ": " + value);
  }
  return parsed;
}

unsigned int parse_unsigned(const char *value, const std::string &name) {
  char *end = nullptr;
  const unsigned long parsed = std::strtoul(value, &end, 10);
  if (end == value || *end != '\0' ||
      parsed > std::numeric_limits<unsigned int>::max()) {
    fail("invalid " + name + ": " + value);
  }
  return static_cast<unsigned int>(parsed);
}

bool parse_bool(const char *value, const std::string &name) {
  const std::string text(value);
  if (text == "1" || text == "true") {
    return true;
  }
  if (text == "0" || text == "false") {
    return false;
  }
  fail("invalid " + name + ": expected 0/1/false/true");
}

Options parse_options(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    if (argument == "--help") {
      std::cout
          << "Usage: hktex2quad_miq --input field.txt --output uv.txt "
             "[--gradient-scale 1] [--stiffness 5] [--iterations 5] "
             "[--local-iterations 5] [--direct-round 0] "
             "[--round-singularities 1]\n";
      std::exit(0);
    }
    if (index + 1 >= argc) {
      fail("missing value after " + argument);
    }
    const char *value = argv[++index];
    if (argument == "--input") {
      options.input = value;
    } else if (argument == "--output") {
      options.output = value;
    } else if (argument == "--gradient-scale") {
      options.gradient_scale = parse_double(value, "gradient scale");
    } else if (argument == "--stiffness") {
      options.stiffness = parse_double(value, "stiffness");
    } else if (argument == "--iterations") {
      options.iterations = parse_unsigned(value, "iterations");
    } else if (argument == "--local-iterations") {
      options.local_iterations = parse_unsigned(value, "local iterations");
    } else if (argument == "--direct-round") {
      options.direct_round = parse_bool(value, "direct round");
    } else if (argument == "--round-singularities") {
      options.round_singularities =
          parse_bool(value, "round singularities");
    } else {
      fail("unknown argument: " + argument);
    }
  }
  if (options.input.empty() || options.output.empty()) {
    fail("--input and --output are required");
  }
  if (!(options.gradient_scale > 0.0)) {
    fail("gradient scale must be positive");
  }
  if (!(options.stiffness >= 0.0)) {
    fail("stiffness must be non-negative");
  }
  return options;
}

void read_input(const std::string &filename, Eigen::MatrixXd &vertices,
                Eigen::MatrixXi &faces, Eigen::MatrixXd &direction_u,
                Eigen::MatrixXd &direction_v,
                Eigen::VectorXd &target_size) {
  std::ifstream stream(filename);
  if (!stream) {
    fail("cannot open MIQ input: " + filename);
  }
  std::string magic;
  stream >> magic;
  if (magic != "HKTEX2QUAD_MIQ_INPUT_V1") {
    fail("unsupported MIQ input format");
  }
  Eigen::Index vertex_count = 0;
  Eigen::Index face_count = 0;
  stream >> vertex_count >> face_count;
  if (!stream || vertex_count <= 0 || face_count <= 0) {
    fail("invalid vertex/face counts in MIQ input");
  }
  vertices.resize(vertex_count, 3);
  faces.resize(face_count, 3);
  direction_u.resize(face_count, 3);
  direction_v.resize(face_count, 3);
  target_size.resize(face_count);
  for (Eigen::Index row = 0; row < vertex_count; ++row) {
    stream >> vertices(row, 0) >> vertices(row, 1) >> vertices(row, 2);
  }
  for (Eigen::Index row = 0; row < face_count; ++row) {
    stream >> faces(row, 0) >> faces(row, 1) >> faces(row, 2);
  }
  for (Eigen::Index row = 0; row < face_count; ++row) {
    stream >> direction_u(row, 0) >> direction_u(row, 1) >>
        direction_u(row, 2) >> direction_v(row, 0) >>
        direction_v(row, 1) >> direction_v(row, 2) >> target_size(row);
  }
  if (!stream) {
    fail("truncated or malformed MIQ input");
  }
  if (!vertices.allFinite() || !direction_u.allFinite() ||
      !direction_v.allFinite() || !target_size.allFinite()) {
    fail("MIQ input contains NaN or infinite values");
  }
  if ((faces.array() < 0).any() ||
      (faces.array() >= vertex_count).any()) {
    fail("MIQ input contains an invalid face index");
  }
  if ((target_size.array() <= 0.0).any()) {
    fail("MIQ target sizes must be positive");
  }
  for (Eigen::Index row = 0; row < face_count; ++row) {
    const double first_norm = direction_u.row(row).norm();
    const double second_norm = direction_v.row(row).norm();
    if (!(first_norm > 1e-14) || !(second_norm > 1e-14)) {
      fail("MIQ input contains a zero direction");
    }
    direction_u.row(row) /= first_norm;
    direction_v.row(row) /= second_norm;
  }
}

void write_output(const std::string &filename, const Eigen::MatrixXd &uv,
                  const Eigen::MatrixXi &face_uv,
                  const Eigen::MatrixXi &seams,
                  const Eigen::MatrixXi &mismatch,
                  const Eigen::VectorXi &singular,
                  const Eigen::VectorXi &singularity_index) {
  std::ofstream stream(filename);
  if (!stream) {
    fail("cannot open MIQ output: " + filename);
  }
  stream << "HKTEX2QUAD_MIQ_OUTPUT_V1\n";
  stream << uv.rows() << ' ' << face_uv.rows() << ' ' << singular.rows()
         << '\n';
  stream << std::setprecision(17);
  for (Eigen::Index row = 0; row < uv.rows(); ++row) {
    // libigl's internal discrete-gradient matrix already accounts for the
    // minus sign in its RHS convention. The returned UV map is right-handed
    // with respect to the supplied (PD1, PD2) frame and must not be reflected.
    stream << uv(row, 0) << ' ' << uv(row, 1) << '\n';
  }
  for (Eigen::Index row = 0; row < face_uv.rows(); ++row) {
    stream << face_uv(row, 0) << ' ' << face_uv(row, 1) << ' '
           << face_uv(row, 2) << '\n';
  }
  for (Eigen::Index row = 0; row < seams.rows(); ++row) {
    stream << seams(row, 0) << ' ' << seams(row, 1) << ' ' << seams(row, 2)
           << '\n';
  }
  for (Eigen::Index row = 0; row < mismatch.rows(); ++row) {
    stream << mismatch(row, 0) << ' ' << mismatch(row, 1) << ' '
           << mismatch(row, 2) << '\n';
  }
  for (Eigen::Index row = 0; row < singular.rows(); ++row) {
    stream << singular(row) << ' ' << singularity_index(row) << '\n';
  }
  if (!stream) {
    fail("failed while writing MIQ output");
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi faces;
    Eigen::MatrixXd direction_u;
    Eigen::MatrixXd direction_v;
    Eigen::VectorXd target_size;
    read_input(options.input, vertices, faces, direction_u, direction_v,
               target_size);

    // PD1/PD2 are per-face Jacobian rows. Their magnitude carries the local
    // inverse target edge length, so MIQ honors HKTex2Quad's size field rather
    // than using only a global quad count.
    Eigen::MatrixXd pd1 = direction_u;
    Eigen::MatrixXd pd2 = direction_v;
    for (Eigen::Index row = 0; row < faces.rows(); ++row) {
      pd1.row(row) /= target_size(row);
      pd2.row(row) /= target_size(row);
    }

    Eigen::MatrixXd bisector_1;
    Eigen::MatrixXd bisector_2;
    igl::compute_frame_field_bisectors(vertices, faces, pd1, pd2, bisector_1,
                                       bisector_2);
    Eigen::MatrixXd combed_bisector_1;
    Eigen::MatrixXd combed_bisector_2;
    igl::comb_cross_field(vertices, faces, bisector_1, bisector_2,
                          combed_bisector_1, combed_bisector_2);

    Eigen::MatrixXi mismatch;
    igl::cross_field_mismatch(vertices, faces, combed_bisector_1,
                              combed_bisector_2, true, mismatch);
    Eigen::VectorXi singular;
    Eigen::VectorXi singularity_index;
    igl::find_cross_field_singularities(vertices, faces, mismatch, singular,
                                        singularity_index);
    Eigen::MatrixXi seams;
    igl::cut_mesh_from_singularities(vertices, faces, mismatch, seams);

    Eigen::MatrixXd combed_pd1;
    Eigen::MatrixXd combed_pd2;
    igl::comb_frame_field(vertices, faces, pd1, pd2, combed_bisector_1,
                          combed_bisector_2, combed_pd1, combed_pd2);

    const double diagonal =
        (vertices.colwise().maxCoeff() - vertices.colwise().minCoeff()).norm();
    if (!(diagonal > 0.0)) {
      fail("input mesh has a zero bounding-box diagonal");
    }
    Eigen::MatrixXd uv;
    Eigen::MatrixXi face_uv;
    igl::copyleft::comiso::miq(
        vertices, faces, combed_pd1, combed_pd2, mismatch, singular, seams, uv,
        face_uv, diagonal * options.gradient_scale, options.stiffness,
        options.direct_round, options.iterations, options.local_iterations,
        true, options.round_singularities);

    if (uv.rows() == 0 || face_uv.rows() != faces.rows() || !uv.allFinite()) {
      fail("MIQ returned an empty or invalid parameterization");
    }
    write_output(options.output, uv, face_uv, seams, mismatch, singular,
                 singularity_index);
    std::cout << "MIQ solved: faces=" << faces.rows() << " uv_vertices="
              << uv.rows() << " singularities=" << singular.sum() << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "hktex2quad_miq: " << error.what() << '\n';
    return 1;
  }
}
