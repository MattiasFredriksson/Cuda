#pragma once

#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <assert.h>

struct SimpleMesh
{
	float _bb[6] = { FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX };
	//Face indices:
	std::vector<unsigned int> _face_ind;
	std::vector<unsigned int> _face_nor;
	std::vector<unsigned int> _face_uv;
	//Vertex data:
	std::vector<float> _vertices;
	std::vector<float> _normals;
	std::vector<float> _uv;
	//Other:
	std::vector<unsigned int> _edges;

	/* Get if this mesh contains minimal set of required data. */
	bool valid();
	void append(SimpleMesh& other);
};


bool readObj(const char *file, SimpleMesh& mesh);
void fitBB(float* bb, float* v);
void mergeBB(float* bb, float* bb2);


#ifdef SIMPLE_MESH

bool SimpleMesh::valid()
{
	if (_vertices.size() == 0 || _face_ind.size() == 0) return false;
	for (int i = 0; i < _face_ind.size(); i++)
	{
		if (_face_ind[i] >= _vertices.size())
			return false;
	}
	for (int i = 0; i < _face_nor.size(); i++)
	{
		if (_face_nor[i] >= _normals.size())
			return false;
	}
	for (int i = 0; i < _face_uv.size(); i++)
	{
		if (_face_uv[i] >= _uv.size())
			return false;
	}
	for (int i = 0; i < _edges.size(); i++)
	{
		if (_edges[i] >= _vertices.size())
			return false;
	}
	return true;
}

void SimpleMesh::append(SimpleMesh& other)
{
	if (!other.valid()) return;
	//Reserve
	_face_ind.reserve(_face_ind.size() + other._face_ind.size());
	_face_nor.reserve(_face_nor.size() + other._face_nor.size());
	_face_uv.reserve(_face_uv.size() + other._face_uv.size());
	_vertices.reserve(_vertices.size() + other._vertices.size());
	_normals.reserve(_normals.size() + other._normals.size());
	_uv.reserve(_uv.size() + other._uv.size());
	_edges.reserve(_edges.size() + other._edges.size());
	// Copy indices
	int offset = _vertices.size() / 3;
	for (int i = 0; i < other._face_ind.size(); i++) _face_ind.push_back(other._face_ind[i] + offset);
	offset = _normals.size() / 3;
	for (int i = 0; i < other._face_nor.size(); i++) _face_nor.push_back(other._face_nor[i] + offset);
	offset = _uv.size() / 3;
	for (int i = 0; i < other._face_uv.size(); i++) _face_uv.push_back(other._face_uv[i] + offset);
	offset = _edges.size() / 3;
	for (int i = 0; i < other._edges.size(); i++) _edges.push_back(other._edges[i] + offset);
	// Copy vertices
	for (int i = 0; i < other._vertices.size(); i++) _vertices.push_back(other._vertices[i]);
	for (int i = 0; i < other._normals.size(); i++) _normals.push_back(other._normals[i]);
	for (int i = 0; i < other._uv.size(); i++) _uv.push_back(other._uv[i]);
	mergeBB(_bb, other._bb);
}

void consumeWhiteSpace(std::stringstream& ss)
{
	char c;
	//Eat whitespace
	while ((c = ss.peek()) ==  ' ' || c == '\t')
		c = ss.get();
}
/* Fit BB with vector. */
void fitBB(float* bb, float* v)
{
	for (int i = 0; i < 3; i++) { bb[i * 2] = std::fminf(v[i], bb[i*2]); bb[i * 2 + 1] = std::fmaxf(v[i], bb[i * 2 + 1]); }
}
/* Fit bb with another bounding box bb2. */
void mergeBB(float* bb, float* bb2)
{
	for (int i = 0; i < 3; i++) { bb[i * 2] = std::fminf(bb2[i*2], bb[i * 2]); bb[i * 2 + 1] = std::fmaxf(bb2[i*2+1], bb[i * 2 + 1]); }
}

void initBB(float* bb, float* v)
{
	for (int i = 0; i < 3; i++) { bb[i*2] = v[i]; bb[i*2+1] = v[i]; }
}

/* Read nodes and faces from an obj file:
*/
 bool readObj(const char *file, SimpleMesh& mesh)
{
	//Open
	std::ifstream stream(file);
	if (stream.is_open())
	{
		//Line params:
		std::string s, head, item, vertex[3];
		std::stringstream ss, ss2;
		//Input vars:
		unsigned int a = 0, b = 0;
		unsigned int v[3], t[2], n[3];
		float xyz[3];
		//Read lines
		while (std::getline(stream, s))
		{
			ss = std::stringstream(s);
			if (!std::getline(ss, head, ' '))
				continue;
			if (head.empty()) continue;
			consumeWhiteSpace(ss);

			//Parse data:
			if (head.length() == 2)
			{
				if (head == "vn") {					//Normals
					ss >> xyz[0]; ss >> xyz[1]; ss >> xyz[2];
					mesh._normals.push_back(xyz[0]);
					mesh._normals.push_back(xyz[1]);
					mesh._normals.push_back(xyz[2]);
				}
				else if (head == "vt")				//UV
				{
					ss >> xyz[0]; ss >> xyz[1];
					mesh._uv.push_back(xyz[0]);
					mesh._uv.push_back(xyz[1]);
				}
			}
			else if (head.length() == 1)
			{
				if (head[0] == 'l') {				//Edges
					ss >> a; ss >> b;
					mesh._edges.push_back(a - 1);//.obj indices are +1
					mesh._edges.push_back(b - 1);
				}
				else if (head[0] == 'v') {			//Vertices
					ss >> xyz[0]; ss >> xyz[1]; ss >> xyz[2];
					if (mesh._vertices.size() == 0) initBB(mesh._bb, xyz);
					else fitBB(mesh._bb, xyz);

					mesh._vertices.push_back(xyz[0]);
					mesh._vertices.push_back(xyz[1]);
					mesh._vertices.push_back(xyz[2]);
				}
				else if (head[0] == 'f')			//Faces
				{
					int num_vert = 0, num_tex = 0, num_norm = 0;
					while (std::getline(ss, item, ' '))
					{
						if (item.empty()) continue;
						int i = 0;
						ss2 = std::stringstream(item);
						for (; i < 3; i++)
						{
							if (!std::getline(ss2, vertex[i], '/'))
								break;
						}

						// Parse: Vertex/UV/Normal
						if (i > 0)
							v[num_vert++] = std::atoi(vertex[0].c_str());
						else
							break; //Terminate this line (face invalid)
						if (i > 1 && !vertex[1].empty())
							t[num_tex++] = std::atoi(vertex[1].c_str());
						if (i > 2 && !vertex[2].empty())
							n[num_norm++] = std::atoi(vertex[2].c_str());
					}

					if (num_vert == 3)
					{
						for (int i = 0; i < 3; i++)
						{
							mesh._face_ind.push_back(v[i] - 1);
							if (num_tex == num_vert)	mesh._face_uv.push_back(t[i] - 1);
							if (num_norm == num_vert)	mesh._face_nor.push_back(n[i] - 1);
							assert(v[i] > 0);
							assert(t[i] > 0);
							assert(n[i] > 0);
						}
					}
					else
						// TODO : Convert N-gons/quads to triangles
						std::cout << "Warning: N-gon/Quad detected only triangles are supported\n";
				}
			}
		}
	}
	else //No nodemap could be read:
	{
		std::cout << "error - Node network file: " << file << " could not be opened\n";
		return false;
	}
	return true;
}

#endif