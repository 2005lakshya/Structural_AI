import axios from 'axios';

let API_BASE_URL = process.env.REACT_APP_API_URL || 'http://127.0.0.1:8000';

export const setApiBaseUrl = (url) => {
  API_BASE_URL = url;
};

export const getApiBaseUrl = () => API_BASE_URL;

const getApi = () => axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Health check
export const healthCheck = async () => {
  try {
    const response = await getApi().get('/health');
    return response.data;
  } catch (error) {
    console.error('Health check failed:', error);
    throw error;
  }
};

export const getModelPerformance = async () => {
  try {
    const response = await getApi().get('/model_performance');
    return response.data;
  } catch (error) {
    console.error('Error fetching model performance:', error);
    throw error;
  }
};

export const analyzeImage = async (file) => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    const response = await axios.post(`${API_BASE_URL}/analyze_image`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  } catch (error) {
    console.error('Error analyzing image:', error);
    throw error;
  }
};

export const analyzeThermal = async (file) => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    const response = await axios.post(`${API_BASE_URL}/analyze_thermal`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  } catch (error) {
    console.error('Error analyzing thermal image:', error);
    throw error;
  }
};

export const predictRisk = async (data) => {
  try {
    const response = await getApi().post('/predict_risk', data);
    return response.data;
  } catch (error) {
    console.error('Error predicting risk:', error);
    throw error;
  }
};

export const explainRisk = async (data) => {
  try {
    const response = await getApi().post('/explain_risk', data);
    return response.data;
  } catch (error) {
    console.error('Error explaining risk:', error);
    throw error;
  }
};

export const analyzeStructure = async (data) => {
  try {
    const response = await getApi().post('/analyze_structure', data);
    return response.data;
  } catch (error) {
    console.error('Error analyzing structure:', error);
    throw error;
  }
};

export const generateReportUrl = () => {
  return `${API_BASE_URL}/generate_report`;
};

export const chatWithInspector = async (data) => {
  try {
    const response = await getApi().post('/chat', data);
    return response.data;
  } catch (error) {
    console.error('Error chatting with inspector:', error);
    throw error;
  }
};

export const calculateCrackWidthApi = async (data) => {
  try {
    const response = await getApi().post('/calculate_crack_width', data);
    return response.data;
  } catch (error) {
    console.error('Error calculating crack width:', error);
    throw error;
  }
};

export const measureCrackWidthFromImage = async (file, params = {}) => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    const query = new URLSearchParams({
      camera_distance_cm: params.camera_distance_cm ?? 50,
      known_length_px: params.known_length_px ?? 0,
      known_length_mm: params.known_length_mm ?? 0,
      dpi: params.dpi ?? 0,
      exposure: params.exposure ?? 'moderate',
      cover_mm: params.cover_mm ?? 40,
      design_life_years: params.design_life_years ?? 50,
    }).toString();
    const response = await axios.post(
      `${API_BASE_URL}/measure_crack_width_image?${query}`,
      formData,
      { headers: { 'Content-Type': 'multipart/form-data' } }
    );
    return response.data;
  } catch (error) {
    console.error('Error measuring crack width from image:', error);
    throw error;
  }
};

export const calculateDurabilityImpactApi = async (data) => {
  try {
    const response = await getApi().post('/calculate_durability_impact', data);
    return response.data;
  } catch (error) {
    console.error('Error calculating durability impact:', error);
    throw error;
  }
};

export const calculateCrackDepthApi = async (data) => {
  try {
    const response = await getApi().post('/calculate_crack_depth', data);
    return response.data;
  } catch (error) {
    console.error('Error calculating crack depth:', error);
    throw error;
  }
};

export const calculateUPVDepthApi = async (data) => {
  try {
    const response = await getApi().post('/calculate_upv_depth', data);
    return response.data;
  } catch (error) {
    console.error('Error calculating UPV depth:', error);
    throw error;
  }
};


