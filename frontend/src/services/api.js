import axios from 'axios';

// When running through Vite dev server, requests are proxied.
// When running in production (served by FastAPI), API is at /api.
const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 45000,
});

export const api = {
  // Health & Status
  getHealth: async () => {
    const res = await apiClient.get('/health');
    return res.data;
  },

  getProjectStatus: async () => {
    const res = await apiClient.get('/project/status');
    return res.data;
  },

  // Dataset
  getDatasetInfo: async () => {
    const res = await apiClient.get('/dataset/info');
    return res.data;
  },

  getDatasetSamples: async () => {
    const res = await apiClient.get('/dataset/samples');
    return res.data;
  },

  // Primary Re-ID Search
  searchQuery: async (file, view = 'auto', topK = 5, excludeSameCamera = false) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('view', view);
    formData.append('top_k', topK);
    formData.append('exclude_same_camera', excludeSameCamera);

    const res = await apiClient.post('/search', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return res.data;
  },

  getSearchResult: async (searchId) => {
    const res = await apiClient.get(`/results/${searchId}`);
    return res.data;
  },

  // Pairwise Cross-View Verification
  compareImages: async (file1, file2, cam1 = 'auto', cam2 = 'auto', threshold = 0.40) => {
    const formData = new FormData();
    formData.append('file1', file1);
    formData.append('file2', file2);
    formData.append('cam1', cam1);
    formData.append('cam2', cam2);
    formData.append('threshold', threshold);

    const res = await apiClient.post('/compare', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return res.data;
  },

  // Model & Architecture
  getModelInfo: async () => {
    const res = await apiClient.get('/model/info');
    return res.data;
  },

  // Evaluation Metrics
  getEvaluation: async () => {
    const res = await apiClient.get('/evaluation');
    return res.data;
  },

  // Image Enhancement
  enhanceImage: async (file, clipLimit = 2.0, gamma = 1.2) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('clip_limit', clipLimit);
    formData.append('gamma', gamma);

    const res = await apiClient.post('/enhance', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return res.data;
  },
};

export default api;

