package main

import (
    "encoding/json"
    "fmt"
    "io"
    "os"
    "reflect"
    "strconv"
    "time"
)

// Il nome della funzione viene sostituito dal GoRunner dopo la sua verifica.
var targetFunction = reflect.ValueOf(__REDATASET_ENTRY_POINT__)

type request struct {
    Index int               `json:"index"`
    Args  []json.RawMessage `json:"args"`
}

type response struct {
    Index int         `json:"index"`
    Value interface{} `json:"value"`
    Error string      `json:"error,omitempty"`
}

func invoke(req request) (resp response) {
    resp.Index = req.Index
    defer func() {
        if recovered := recover(); recovered != nil {
            resp.Error = fmt.Sprint(recovered)
        }
    }()

    kind := targetFunction.Kind()
    if kind != reflect.Func {
        resp.Error = "entry point non richiamabile"
        return
    }
    signature := targetFunction.Type()
    if signature.IsVariadic() || signature.NumIn() != len(req.Args) || signature.NumOut() != 1 {
        resp.Error = "firma Go non supportata: sono richiesti argomenti fissi e un solo output"
        return
    }

    args := make([]reflect.Value, len(req.Args))
    for index, encoded := range req.Args {
        // Decodifica direttamente nel tipo dichiarato dalla funzione Go.
        holder := reflect.New(signature.In(index))
        if err := json.Unmarshal(encoded, holder.Interface()); err != nil {
            resp.Error = fmt.Sprintf("argomento %d: %v", index, err)
            return
        }
        args[index] = holder.Elem()
    }

    result := targetFunction.Call(args)[0].Interface()
    // Verifica che l'output possa attraversare il protocollo JSON.
    if _, err := json.Marshal(result); err != nil {
        resp.Error = fmt.Sprintf("output non serializzabile: %v", err)
        return
    }
    resp.Value = result
    return
}

const timerResultPrefix = "__SPEC2FAAS_TIMER_V1__"

// Nella modalita' Timer, ogni template viene deserializzato a ogni invocazione.
// Il contatore e la durata sono emessi solo dopo la finestra di lavoro.
func runTimer(seconds float64, maxInvocations int) error {
    decoder := json.NewDecoder(os.Stdin)
    encoder := json.NewEncoder(os.Stdout)
    templates := make([]json.RawMessage, 0)
    for {
        var raw json.RawMessage
        if err := decoder.Decode(&raw); err != nil {
            if err == io.EOF {
                break
            }
            return fmt.Errorf("Timer: input JSON non valido: %w", err)
        }
        var req request
        if err := json.Unmarshal(raw, &req); err != nil || req.Index != len(templates) {
            return fmt.Errorf("Timer: indice o template JSON non valido")
        }
        templates = append(templates, raw)
    }
    if len(templates) == 0 {
        return fmt.Errorf("Timer: almeno un caso JSON e' necessario")
    }
    started := time.Now()
    count := 0
    for count < maxInvocations {
        var req request
        if err := json.Unmarshal(templates[count%len(templates)], &req); err != nil {
            return fmt.Errorf("Timer: JSON non valido: %w", err)
        }
        req.Index = count
        resp := invoke(req)
        if resp.Error != "" {
            return fmt.Errorf("Timer: funzione Go fallita: %s", resp.Error)
        }
        if err := encoder.Encode(resp); err != nil {
            return fmt.Errorf("Timer: output JSON non valido: %w", err)
        }
        count++
        if time.Since(started).Seconds() >= seconds {
            break
        }
    }
    elapsed := time.Since(started).Seconds()
    reason := "duration_reached"
    if elapsed < seconds {
        reason = "invocation_cap_reached"
    }
    result := struct {
        Invocations int `json:"invocations"`
        WorkloadSeconds float64 `json:"workload_seconds"`
        StopReason string `json:"stop_reason"`
    }{count, elapsed, reason}
    encoded, err := json.Marshal(result)
    if err != nil {
        return err
    }
    fmt.Fprintln(os.Stderr, timerResultPrefix+string(encoded))
    return nil
}

func main() {
    if len(os.Args) != 1 {
        if len(os.Args) != 5 || os.Args[1] != "--timer-seconds" || os.Args[3] != "--timer-max-invocations" {
            fmt.Fprintln(os.Stderr, "Parametri GoHarness non validi")
            os.Exit(2)
        }
        seconds, err := strconv.ParseFloat(os.Args[2], 64)
        if err != nil || seconds <= 0 || seconds != seconds || seconds > 1e9 {
            fmt.Fprintln(os.Stderr, "Durata Timer non valida")
            os.Exit(2)
        }
        maximum, err := strconv.Atoi(os.Args[4])
        if err != nil || maximum < 1 {
            fmt.Fprintln(os.Stderr, "Cap Timer non valido")
            os.Exit(2)
        }
        if err := runTimer(seconds, maximum); err != nil {
            fmt.Fprintln(os.Stderr, err)
            os.Exit(2)
        }
        return
    }
    decoder := json.NewDecoder(os.Stdin)
    encoder := json.NewEncoder(os.Stdout)
    for {
        var req request
        if err := decoder.Decode(&req); err != nil {
            if err == io.EOF {
                return
            }
            fmt.Fprintln(os.Stderr, "input JSON non valido:", err)
            os.Exit(2)
        }
        if err := encoder.Encode(invoke(req)); err != nil {
            fmt.Fprintln(os.Stderr, "output JSON non valido:", err)
            os.Exit(2)
        }
    }
}
