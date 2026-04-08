{{/*
Common labels
*/}}
{{- define "rag.labels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end }}

{{/*
API selector labels
*/}}
{{- define "rag.api.selectorLabels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}-api
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
vLLM selector labels
*/}}
{{- define "rag.vllm.selectorLabels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}-vllm
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Qdrant selector labels
*/}}
{{- define "rag.qdrant.selectorLabels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}-qdrant
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Full name helper
*/}}
{{- define "rag.fullname" -}}
{{ .Release.Name }}-{{ .Chart.Name }}
{{- end }}
