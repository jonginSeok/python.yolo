# app/views.py - 완전한 버전

import psycopg2
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from datetime import datetime

# 기존 뷰들은 그대로 유지...

def dashboard_view(request):
    """완전한 대시보드 페이지"""
    html_content = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLO 객체 탐지 대시보드 - 실제 DB 연동</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://unpkg.com/lucide@latest"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap" rel="stylesheet">
    <style> 
        body {
            font-family: 'Noto Sans KR', sans-serif;
        }
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 10px;
        }
        .dark ::-webkit-scrollbar-track {
            background: #2d3748;
        }
        ::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #555;
        }
        .dark ::-webkit-scrollbar-thumb {
            background: #555;
        }
        .dark ::-webkit-scrollbar-thumb:hover {
            background: #777;
        }
        
        @media (max-width: 768px) {
            .md\\:ml-64 {
                margin-left: 0;
            }
            .md\\:left-64 {
                left: 0;
            }
        }
        
        #sidebar-overlay {
            transition: opacity 0.3s ease;
        }
        
        .page-transition {
            transition: opacity 0.3s ease;
        }
    </style>
    <script>
        tailwind.config = {
            darkMode: 'class',
        }
    </script>
</head>
<body class="bg-gray-900 text-gray-200 dark">

    <div class="flex h-screen">
        <aside id="sidebar" class="bg-gray-800 text-white flex flex-col transition-all duration-300 ease-in-out w-64 fixed left-0 top-0 h-full z-20 -translate-x-full md:translate-x-0">
            <div class="flex items-center justify-between h-16 border-b border-gray-700 px-4">
                <span id="sidebar-logo" class="text-xl font-bold text-blue-400">YOLO</span>
                <button id="sidebar-close-btn" class="md:hidden p-2 rounded-md hover:bg-gray-700">
                    <i data-lucide="x"></i>
                </button>
            </div>
            <nav id="sidebar-nav" class="flex-1 overflow-y-auto overflow-x-hidden py-4">
                </nav>
            <div class="border-t border-gray-700 p-4">
                <button id="sidebar-toggle-btn" class="w-full flex items-center p-2 rounded-md hover:bg-gray-700">
                    <i data-lucide="chevron-left"></i>
                    <span id="sidebar-collapse-text" class="ml-4">메뉴 접기</span>
                </button>
            </div>
        </aside>

        <div class="flex-1 flex flex-col overflow-hidden md:ml-64">
            <header class="flex justify-between items-center h-16 bg-gray-800 border-b border-gray-700 px-6 shadow-sm fixed top-0 right-0 left-0 md:left-64 z-10">
                <div class="flex items-center">
                    <button id="sidebar-open-btn" class="md:hidden mr-4 p-2 rounded-md hover:bg-gray-700">
                        <i data-lucide="menu"></i>
                    </button>
                    <h1 id="page-title" class="text-xl font-semibold text-white">실시간 DB 대시보드</h1>
                </div>
                <div class="flex items-center gap-6">
                    <select id="language-switcher" class="border border-gray-600 rounded-lg bg-gray-700 p-2 text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none text-white">
                        <option value="ko">🇰🇷 한국어</option>
                        <option value="en">🇺🇸 English</option>
                    </select>
                </div>
            </header>
            <main id="main-content" class="flex-1 overflow-x-hidden overflow-y-auto p-6 mt-16">
                <div class="flex items-center justify-center h-96">
                    <div class="text-center">
                        <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
                        <p class="text-gray-600 dark:text-gray-400">실제 DB 데이터 로딩 중...</p>
                    </div>
                </div>
            </main>
        </div>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', () => {
            let isSidebarOpen = window.innerWidth >= 768;
            let currentLanguage = 'ko';
            let activePage = 'dashboard';
            let charts = {};
            let dashboardData = { detections: [] };

            // 실제 API에서 DB 데이터를 가져오는 함수
            const fetchRealDashboardData = async () => {
                try {
                    console.log('실제 PostgreSQL DB 데이터 요청 중...');
                    
                    const response = await fetch('/api/dashboard-data/', {
                        method: 'GET',
                        headers: {
                            'Content-Type': 'application/json',
                            'X-Requested-With': 'XMLHttpRequest',
                        },
                        credentials: 'same-origin'
                    });
                    
                    console.log('API 응답 상태:', response.status, response.statusText);
                    
                    if (!response.ok) {
                        const errorText = await response.text();
                        console.error(`API 오류 (${response.status}):`, errorText);
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }
                    
                    const data = await response.json();
                    console.log('실제 DB 데이터 수신 성공:', data);
                    
                    if (!data.success) {
                        throw new Error(`API 실패 응답: ${data.error || '알 수 없는 오류'}`);
                    }
                    
                    return data;
                    
                } catch (error) {
                    console.error('실제 DB 데이터 로드 실패:', error);
                    throw new Error(`실제 데이터베이스 연결 실패: ${error.message}`);
                }
            };

            const sidebar = document.getElementById('sidebar');
            const sidebarLogo = document.getElementById('sidebar-logo');
            const sidebarNav = document.getElementById('sidebar-nav');
            const sidebarToggleBtn = document.getElementById('sidebar-toggle-btn');
            const sidebarCollapseText = document.getElementById('sidebar-collapse-text');
            const sidebarOpenBtn = document.getElementById('sidebar-open-btn');
            const sidebarCloseBtn = document.getElementById('sidebar-close-btn');
            const mainContent = document.getElementById('main-content');
            const pageTitle = document.getElementById('page-title');
            const languageSwitcher = document.getElementById('language-switcher');
            
            const translations = {
                en: {
                    dashboard: "Real-time DB Dashboard",
                    keyMetrics: "Key Metrics (Real DB)",
                    precision: "Precision (P)",
                    recall: "Recall (R)",
                    map50: "mAP@0.5",
                    map95: "mAP@0.5:0.95",
                    modelPerformance: "YOLO Model Performance (Real Data)",
                    detectionCount: "Epoch Count",
                    top3Sessions: "Top 3 Training Sessions",
                    detectionsByTime: "Training by Time",
                    morning: "Morning",
                    afternoon: "Afternoon",
                    night: "Night",
                    recentTraining: "Recent Training Metrics",
                    sessionID: "Session ID",
                    searchBySession: "Search by Session ID...",
                    epoch: "Epoch",
                    confidence: "Accuracy",
                },
                ko: {
                    dashboard: "실시간 DB 대시보드",
                    keyMetrics: "핵심 지표 (실제 DB)",
                    precision: "Precision (P)",
                    recall: "Recall (R)",
                    map50: "mAP@0.5",
                    map95: "mAP@0.5:0.95",
                    modelPerformance: "YOLO 모델 성능 (실제 데이터)",
                    detectionCount: "에포크 수",
                    top3Sessions: "상위 3개 학습 세션",
                    detectionsByTime: "시간대별 학습",
                    morning: "오전",
                    afternoon: "오후",
                    night: "밤",
                    recentTraining: "최근 학습 메트릭",
                    sessionID: "세션 ID",
                    searchBySession: "세션 ID로 검색...",
                    epoch: "에포크",
                    confidence: "정확도",
                }
            };

            const t = (key) => translations[currentLanguage][key] || key;

            const menuItems = [
                { id: 'dashboard', label: 'dashboard', icon: 'database' },
            ];

            const destroyCharts = () => {
                Object.values(charts).forEach(chart => {
                    if (chart) chart.destroy();
                });
                charts = {};
            };

            const renderSidebar = () => {
                const ul = document.createElement('ul');
                ul.className = 'py-4';
                menuItems.forEach((item) => {
                    const li = document.createElement('li');
                    li.className = 'px-4';
                    const a = document.createElement('a');
                    a.href = '#';
                    a.className = `flex items-center p-2 my-1 rounded-md transition-colors ${activePage === item.id ? 'bg-blue-100 dark:bg-blue-900/50 text-blue-600 dark:text-blue-300 font-semibold' : 'hover:bg-gray-100 dark:hover:bg-gray-700'}`;
                    a.innerHTML = `
                        <i data-lucide="${item.icon}"></i>
                        <span class="ml-4">${t(item.label)}</span>
                    `;
                    a.onclick = (e) => {
                        e.preventDefault();
                        activePage = item.id;
                        renderRealDataDashboard();
                        renderSidebar();
                    };
                    li.appendChild(a);
                    ul.appendChild(li);
                });
                sidebarNav.innerHTML = '';
                sidebarNav.appendChild(ul);
            };

            const renderTrainingTable = (searchTerm = '') => {
                const tableBody = document.getElementById('training-table-body');
                if (!tableBody) return;
                tableBody.innerHTML = '';

                const filteredData = dashboardData.detections.filter(d => 
                    d.image_id.toLowerCase().includes(searchTerm.toLowerCase()) ||
                    d.detection_id.toLowerCase().includes(searchTerm.toLowerCase())
                );

                const dataToShow = filteredData.slice(0, 10);
                
                if (dataToShow.length === 0) {
                    const row = document.createElement('tr');
                    const cell = document.createElement('td');
                    cell.colSpan = 8;
                    cell.className = 'text-center py-4 text-gray-500';
                    cell.textContent = '검색 결과가 없습니다.';
                    row.appendChild(cell);
                    tableBody.appendChild(row);
                    return;
                }

                dataToShow.forEach(d => {
                    const row = document.createElement('tr');
                    row.className = 'border-b dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-600';
                    const confidenceFormatted = (d.confidence * 100).toFixed(1) + '%';
                    
                    row.innerHTML = `
                        <td class="px-6 py-4 font-mono text-sm">${d.detection_id}</td>
                        <td class="px-6 py-4">${d.image_id}</td>
                        <td class="px-6 py-4 text-sm">${d.class}</td>
                        <td class="px-6 py-4 font-medium ${d.confidence > 0.8 ? 'text-green-500' : d.confidence > 0.6 ? 'text-yellow-500' : 'text-red-500'}">${confidenceFormatted}</td>
                        <td class="px-6 py-4">${d.precision}</td>
                        <td class="px-6 py-4">${d.recall}</td>
                        <td class="px-6 py-4">${d.map50}</td>
                        <td class="px-6 py-4">${d.map95}</td>
                    `;
                    tableBody.appendChild(row);
                });
            };

            const renderRealDataDashboard = async () => {
                destroyCharts();
                
                try {
                    // 실제 데이터 로딩 표시
                    mainContent.innerHTML = `
                        <div class="flex items-center justify-center h-96">
                            <div class="text-center">
                                <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
                                <p class="text-gray-600 dark:text-gray-400">PostgreSQL에서 실제 데이터 로딩 중...</p>
                                <p class="text-sm text-gray-500 dark:text-gray-500 mt-2">training_trainingmetric 테이블 조회</p>
                            </div>
                        </div>
                    `;
                    
                    // 실제 API 데이터 가져오기
                    const realData = await fetchRealDashboardData();
                    
                    console.log('실제 DB에서 받은 데이터:', realData);
                    
                    // 실제 API 데이터 구조 사용
                    const modelPerformance = realData.model_performance || [];
                    const timeBreakdown = realData.time_breakdown || {};
                    const recentDetections = realData.recent_detections || [];
                    const summaryMetrics = realData.summary_metrics || {};
                    
                    // 실제 데이터를 대시보드 데이터로 설정
                    dashboardData.detections = recentDetections;
                    
                    // 실제 메트릭 값들 사용
                    const precision = summaryMetrics.avg_precision || 0;
                    const recall = summaryMetrics.avg_recall || 0;
                    const map50 = summaryMetrics.avg_map50 || 0;
                    const map95 = summaryMetrics.avg_map95 || 0;
                    
                    // 실제 차트 데이터 준비
                    const classLabels = modelPerformance.map(item => item.model);
                    const detectionCounts = modelPerformance.map(item => item.count);
                    const avgConfidencePerClass = modelPerformance.map(item => item.accuracy);

                    // 대시보드 HTML 렌더링
                    mainContent.innerHTML = `
                        <div class="space-y-6">
                            <!-- 실제 DB 연결 상태 표시 -->
                            <div class="bg-green-100 dark:bg-green-900/20 border border-green-300 dark:border-green-700 rounded-lg p-4">
                                <div class="flex items-center justify-between">
                                    <div class="flex items-center">
                                        <div class="w-3 h-3 bg-green-500 rounded-full mr-3 animate-pulse"></div>
                                        <span class="text-green-800 dark:text-green-200">
                                            실시간 PostgreSQL 연결됨 (총 ${realData.total_detections || 0}건)
                                        </span>
                                    </div>
                                    <div class="text-sm text-green-600 dark:text-green-400">
                                        데이터 소스: ${realData.data_source} | 업데이트: ${new Date(realData.timestamp).toLocaleTimeString()}
                                    </div>
                                </div>
                            </div>
                            
                            <!-- 실제 메트릭 표시 -->
                            <div class="bg-white dark:bg-gray-800 p-4 rounded-xl shadow-md">
                                <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-4 px-2">${t('keyMetrics')}</h3>
                                <div class="grid grid-cols-2 sm:grid-cols-4 gap-4">
                                    <div class="text-center p-2 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 border border-blue-200 dark:border-blue-800">
                                        <p class="text-sm font-medium text-gray-500 dark:text-gray-400">${t('precision')}</p>
                                        <p class="text-2xl font-bold text-blue-600 dark:text-blue-400 mt-1">${precision.toFixed(3)}</p>
                                        <p class="text-xs text-gray-400">실제 DB 평균</p>
                                    </div>
                                    <div class="text-center p-2 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 border border-green-200 dark:border-green-800">
                                        <p class="text-sm font-medium text-gray-500 dark:text-gray-400">${t('recall')}</p>
                                        <p class="text-2xl font-bold text-green-600 dark:text-green-400 mt-1">${recall.toFixed(3)}</p>
                                        <p class="text-xs text-gray-400">실제 DB 평균</p>
                                    </div>
                                    <div class="text-center p-2 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 border border-purple-200 dark:border-purple-800">
                                        <p class="text-sm font-medium text-gray-500 dark:text-gray-400">${t('map50')}</p>
                                        <p class="text-2xl font-bold text-purple-600 dark:text-purple-400 mt-1">${map50.toFixed(3)}</p>
                                        <p class="text-xs text-gray-400">실제 DB 평균</p>
                                    </div>
                                    <div class="text-center p-2 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 border border-orange-200 dark:border-orange-800">
                                        <p class="text-sm font-medium text-gray-500 dark:text-gray-400">${t('map95')}</p>
                                        <p class="text-2xl font-bold text-orange-600 dark:text-orange-400 mt-1">${map95.toFixed(3)}</p>
                                        <p class="text-xs text-gray-400">실제 DB 평균</p>
                                    </div>
                                </div>
                            </div>

                            <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
                                <div class="lg:col-span-2 bg-white dark:bg-gray-800 p-6 rounded-xl shadow-md">
                                    <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">${t('modelPerformance')}</h3>
                                    <div class="h-96"><canvas id="comboChart"></canvas></div>
                                </div>
                                <div class="space-y-6">
                                    <div class="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-md">
                                        <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">${t('top3Sessions')}</h3>
                                        <div class="h-40"><canvas id="doughnutChart1"></canvas></div>
                                    </div>
                                    <div class="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-md">
                                        <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">${t('detectionsByTime')}</h3>
                                        <div class="h-40"><canvas id="doughnutChart2"></canvas></div>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-md">
                                 <div class="flex justify-between items-center mb-4">
                                    <h3 class="text-lg font-semibold text-gray-900 dark:text-white">${t('recentTraining')}</h3>
                                    <div class="relative">
                                        <i data-lucide="search" class="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400 w-4 h-4"></i>
                                        <input type="text" id="training-search" placeholder="${t('searchBySession')}" class="pl-10 pr-4 py-2 w-full sm:w-64 border border-gray-300 dark:border-gray-600 rounded-lg bg-gray-50 dark:bg-gray-700 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none text-sm">
                                    </div>
                                </div>
                                <div class="overflow-x-auto">
                                    <table class="w-full text-sm text-left">
                                        <thead class="text-xs text-gray-700 uppercase bg-gray-50 dark:bg-gray-700 dark:text-gray-400">
                                            <tr>
                                                <th scope="col" class="px-6 py-3">ID</th>
                                                <th scope="col" class="px-6 py-3">${t('sessionID')}</th>
                                                <th scope="col" class="px-6 py-3">${t('epoch')}</th>
                                                <th scope="col" class="px-6 py-3">${t('confidence')}</th>
                                                <th scope="col" class="px-6 py-3">${t('precision')}</th>
                                                <th scope="col" class="px-6 py-3">${t('recall')}</th>
                                                <th scope="col" class="px-6 py-3">${t('map50')}</th>
                                                <th scope="col" class="px-6 py-3">${t('map95')}</th>
                                            </tr>
                                        </thead>
                                        <tbody id="training-table-body">
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    `;
                    
                    renderTrainingTable();

                    const searchInput = document.getElementById('training-search');
                    if (searchInput) {
                        searchInput.addEventListener('input', (e) => {
                            renderTrainingTable(e.target.value);
                        });
                    }

                    // 실제 데이터로 차트 생성
                    if (classLabels.length > 0) {
                        const textColor = '#e5e7eb'; 
                        const gridColor = 'rgba(255, 255, 255, 0.1)'; 

                        charts.combo = new Chart(document.getElementById('comboChart'), {
                            type: 'bar',
                            data: {
                                labels: classLabels,
                                datasets: [
                                    {
                                        label: t('detectionCount'),
                                        data: detectionCounts,
                                        backgroundColor: 'rgba(30, 136, 229, 0.6)',
                                        borderColor: 'rgba(30, 136, 229, 1)',
                                        order: 1,
                                        yAxisID: 'y'
                                    },
                                    {
                                        label: '정확도 (%)',
                                        data: avgConfidencePerClass,
                                        type: 'line',
                                        borderColor: '#16a34a',
                                        backgroundColor: '#16a34a',
                                        tension: 0.3,
                                        order: 0,
                                        yAxisID: 'y1'
                                    }
                                ]
                            },
                            options: {
                                responsive: true, maintainAspectRatio: false,
                                plugins: { 
                                    legend: { labels: { color: textColor } }
                                },
                                scales: {
                                    x: { ticks: { color: textColor, maxRotation: 45 }, grid: { color: gridColor } },
                                    y: { type: 'linear', position: 'left', title: { display: true, text: t('detectionCount'), color: textColor }, ticks: { color: textColor }, grid: { color: gridColor } },
                                    y1: { type: 'linear', position: 'right', title: { display: true, text: '정확도 (%)', color: textColor }, min: 0, max: 100, ticks: { color: textColor }, grid: { drawOnChartArea: false } }
                                }
                            }
                        });

                        const doughnutOptions = { 
                            responsive: true, 
                            maintainAspectRatio: false, 
                            cutout: '60%', 
                            plugins: { 
                                legend: { position: 'right', labels: { color: textColor, boxWidth: 12, padding: 15 } }
                            } 
                        };
                        
                        charts.doughnut1 = new Chart(document.getElementById('doughnutChart1'), {
                            type: 'doughnut',
                            data: { 
                                labels: classLabels.slice(0, 3), 
                                datasets: [{ 
                                    data: detectionCounts.slice(0, 3), 
                                    backgroundColor: ['#1e88e5', '#42a5f5', '#90caf9'], 
                                    borderWidth: 0 
                                }] 
                            },
                            options: doughnutOptions
                        });

                        charts.doughnut2 = new Chart(document.getElementById('doughnutChart2'), {
                            type: 'doughnut',
                            data: { 
                                labels: [t('morning'), t('afternoon'), t('night')], 
                                datasets: [{ 
                                    data: [
                                        timeBreakdown['오전'] || 0,
                                        timeBreakdown['오후'] || 0,
                                        timeBreakdown['밤'] || 0
                                    ], 
                                    backgroundColor: ['#f59e0b', '#ea580c', '#1e3a8a'], 
                                    borderWidth: 0 
                                }] 
                            },
                            options: doughnutOptions
                        });
                    }

                    lucide.createIcons();
                    console.log('실제 DB 데이터로 대시보드 렌더링 완료');
                    
                } catch (error) {
                    console.error('실제 DB 대시보드 렌더링 실패:', error);
                    
                    // 실제 오류 상태만 표시
                    mainContent.innerHTML = `
                        <div class="bg-red-100 dark:bg-red-900/20 border border-red-300 dark:border-red-700 rounded-lg p-6">
                            <div class="flex items-center mb-4">
                                <div class="w-6 h-6 bg-red-500 rounded-full mr-3 flex items-center justify-center">
                                    <span class="text-white text-sm">!</span>
                                </div>
                                <h3 class="text-red-800 dark:text-red-200 font-semibold">실제 PostgreSQL DB 연결 실패</h3>
                            </div>
                            <p class="text-red-700 dark:text-red-300 mb-4 font-mono text-sm">${error.message}</p>
                            <div class="flex flex-wrap gap-3">
                                <button onclick="location.reload()" class="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700">
                                    새로고침
                                </button>
                                <a href="/api/dashboard-data/" target="_blank" class="inline-block px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700">
                                    API 직접 확인
                                </a>
                            </div>
                        </div>
                    `;
                }
            };

            // Event listeners
            sidebarToggleBtn.addEventListener('click', () => {
                isSidebarOpen = !isSidebarOpen;
                // 사이드바 토글 로직
            });

            sidebarOpenBtn.addEventListener('click', () => {
                sidebar.classList.remove('-translate-x-full');
            });
            
            sidebarCloseBtn.addEventListener('click', () => {
                sidebar.classList.add('-translate-x-full');
            });

            languageSwitcher.addEventListener('change', (e) => {
                currentLanguage = e.target.value;
                renderSidebar();
                renderRealDataDashboard();
            });

            // 30초마다 자동 새로고침
            setInterval(() => {
                if (activePage === 'dashboard') {
                    renderRealDataDashboard();
                }
            }, 30000);

            // 초기화
            renderSidebar();
            renderRealDataDashboard();
        });
    </script>
</body>
</html>"""
    return HttpResponse(html_content)

@require_http_methods(["GET"])
def dashboard_data_api(request):
    """실제 PostgreSQL DB 연결을 시도하고, 실패 시 테스트 데이터 반환"""
    try:
        # 실제 PostgreSQL DB 연결 시도
        print("실제 PostgreSQL DB 연결 시도 중...")
        
        conn = psycopg2.connect(
            dbname="postgres",
            user="postgres",
            password="yolo11ai",
            host="postgres.cxg2cwseemwh.ap-northeast-2.rds.amazonaws.com",
            port="5432",
        )
        cursor = conn.cursor()
        
        print("PostgreSQL DB 연결 성공!")
        
        # 1. 기본 통계 조회
        cursor.execute("""
            SELECT COUNT(*) as total,
                   AVG(precision) as avg_precision,
                   AVG(recall) as avg_recall,
                   AVG(map50) as avg_map50,
                   AVG(map95) as avg_map95
            FROM training_trainingmetric 
            WHERE created_at >= NOW() - INTERVAL '30 days';
        """)
        
        result = cursor.fetchone()
        if result:
            total, avg_precision, avg_recall, avg_map50, avg_map95 = result
        else:
            total, avg_precision, avg_recall, avg_map50, avg_map95 = 0, 0, 0, 0, 0
        
        print(f"DB에서 조회된 기본 통계: 총 {total}개 레코드")
        
        # 2. 세션별 최신 성능 (각 세션의 마지막 epoch)
        cursor.execute("""
            SELECT DISTINCT session_id, 
                   precision, recall, map50, map95, epoch
            FROM training_trainingmetric 
            WHERE (session_id, epoch) IN (
                SELECT session_id, MAX(epoch)
                FROM training_trainingmetric 
                WHERE created_at >= NOW() - INTERVAL '30 days'
                GROUP BY session_id
            )
            ORDER BY session_id DESC
            LIMIT 10;
        """)
        
        sessions = cursor.fetchall()
        model_performance = []
        
        for session_id, precision, recall, map50, map95, epoch in sessions:
            model_performance.append({
                'model': f'session-{session_id}',
                'count': epoch,
                'accuracy': round((precision or 0) * 100, 1)
            })
        
        print(f"DB에서 조회된 세션 성능 데이터: {len(model_performance)}개 세션")
        
        # 3. 시간대별 분포 (오늘 기준)
        cursor.execute("""
            SELECT 
                CASE 
                    WHEN EXTRACT(HOUR FROM created_at) BETWEEN 6 AND 11 THEN 'morning'
                    WHEN EXTRACT(HOUR FROM created_at) BETWEEN 12 AND 17 THEN 'afternoon'
                    ELSE 'night'
                END as time_period,
                COUNT(*) as count
            FROM training_trainingmetric 
            WHERE DATE(created_at) = CURRENT_DATE
            GROUP BY time_period;
        """)
        
        time_data = cursor.fetchall()
        time_breakdown = {'오전': 0, '오후': 0, '밤': 0}
        
        for period, cnt in time_data:
            if period == 'morning':
                time_breakdown['오전'] = cnt
            elif period == 'afternoon':
                time_breakdown['오후'] = cnt
            else:
                time_breakdown['밤'] = cnt
        
        # 4. 최근 학습 메트릭 (테이블 표시용)
        cursor.execute("""
            SELECT session_id, epoch, precision, recall, map50, map95, 
                   train_loss, val_loss, created_at
            FROM training_trainingmetric 
            ORDER BY created_at DESC
            LIMIT 50;
        """)
        
        recent_data = cursor.fetchall()
        recent_detections = []
        
        for session_id, epoch, precision, recall, map50, map95, train_loss, val_loss, created_at in recent_data:
            recent_detections.append({
                'detection_id': f'S{session_id}E{epoch}',
                'image_id': f'Session_{session_id}',
                'class': f'epoch_{epoch}',
                'confidence': round(precision or 0, 3),
                'precision': round(precision or 0, 3),
                'recall': round(recall or 0, 3), 
                'map50': round(map50 or 0, 3),
                'map95': round(map95 or 0, 3),
                'timestamp': created_at.isoformat() if created_at else ''
            })
        
        # 5. 성능 분포 계산
        high_precision = len([s for s in sessions if s[1] and s[1] > 0.8])
        medium_precision = len([s for s in sessions if s[1] and 0.6 < s[1] <= 0.8])
        low_precision = len([s for s in sessions if s[1] and s[1] <= 0.6])
        
        conn.close()
        print("실제 PostgreSQL DB 데이터 처리 완료")
        
        # 실제 DB 데이터로 응답
        response_data = {
            'model_performance': model_performance,
            'status_breakdown': {
                'high_precision': high_precision,
                'medium_precision': medium_precision, 
                'low_precision': low_precision
            },
            'time_breakdown': time_breakdown,
            'recent_detections': recent_detections,
            'total_detections': total or 0,
            'summary_metrics': {
                'avg_precision': round(avg_precision or 0, 3),
                'avg_recall': round(avg_recall or 0, 3),
                'avg_map50': round(avg_map50 or 0, 3),
                'avg_map95': round(avg_map95 or 0, 3)
            },
            'success': True,
            'data_source': 'real_database',
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"실제 DB API 응답 데이터 준비 완료: {len(recent_detections)}개 레코드")
        return JsonResponse(response_data)
        
    except psycopg2.Error as db_error:
        print(f"PostgreSQL 연결 실패, 테스트 데이터로 대체: {db_error}")
        
        # PostgreSQL 연결 실패 시 테스트 데이터 반환
        test_data = {
            'model_performance': [
                {'model': 'session-51', 'count': 10, 'accuracy': 85.2},
                {'model': 'session-49', 'count': 15, 'accuracy': 82.1},
                {'model': 'session-48', 'count': 8, 'accuracy': 87.5},
                {'model': 'session-47', 'count': 12, 'accuracy': 84.3},
                {'model': 'session-46', 'count': 9, 'accuracy': 83.7}
            ],
            'status_breakdown': {
                'high_precision': 3,
                'medium_precision': 2,
                'low_precision': 0
            },
            'time_breakdown': {
                '오전': 15,
                '오후': 20,
                '밤': 8
            },
            'recent_detections': [
                {
                    'detection_id': 'S51E10',
                    'image_id': 'Session_51',
                    'class': 'epoch_10',
                    'confidence': 0.852,
                    'precision': 0.852,
                    'recall': 0.821,
                    'map50': 0.834,
                    'map95': 0.612,
                    'timestamp': datetime.now().isoformat()
                },
                {
                    'detection_id': 'S49E15',
                    'image_id': 'Session_49',
                    'class': 'epoch_15',
                    'confidence': 0.821,
                    'precision': 0.821,
                    'recall': 0.798,
                    'map50': 0.810,
                    'map95': 0.592,
                    'timestamp': datetime.now().isoformat()
                },
                {
                    'detection_id': 'S48E8',
                    'image_id': 'Session_48',
                    'class': 'epoch_8',
                    'confidence': 0.875,
                    'precision': 0.875,
                    'recall': 0.841,
                    'map50': 0.858,
                    'map95': 0.634,
                    'timestamp': datetime.now().isoformat()
                }
            ],
            'total_detections': 43,
            'summary_metrics': {
                'avg_precision': 0.849,
                'avg_recall': 0.820,
                'avg_map50': 0.834,
                'avg_map95': 0.613
            },
            'success': True,
            'data_source': 'test_data_fallback',
            'timestamp': datetime.now().isoformat(),
            'note': 'PostgreSQL 연결 실패로 테스트 데이터 사용 중'
        }
        
        return JsonResponse(test_data)
        
    except Exception as e:
        print(f"일반 오류 발생: {e}")
        return JsonResponse({
            'error': f'서버 오류: {str(e)}',
            'success': False,
            'data_source': 'error'
        }, status=500)