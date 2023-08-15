class BayesFilter:
    def __init__(self, initial_prob, process_var, measurement_var):
        self.prob = initial_prob
        self.process_var = process_var
        self.measurement_var = measurement_var
        self.prob_uncertainty = 0.5
        
    def predict(self):
        # In diesem Fall bleibt die Vorhersage gleich, da wir kein explizites Bewegungsmodell haben.
        # Nur die Unsicherheit erhöht sich aufgrund des Prozessrauschens.
        self.prob_uncertainty += self.process_var
        
    def update(self, measurement):
        # Gewichtungsfaktor (Kalman-Gain)
        kalman_gain = self.prob_uncertainty / (self.prob_uncertainty + self.measurement_var)
        
        # Aktualisieren Sie die Schätzung mit dem Messwert
        self.prob = self.prob + kalman_gain * (measurement - self.prob)
        
        # Aktualisieren der Unsicherheit
        self.prob_uncertainty = (1 - kalman_gain) * self.prob_uncertainty
        
    def process(self, measurement):
        self.predict()
        self.update(measurement)
        return self.prob


def demo_bayes_filter():
    # Erstellen Sie den Bayes-Filter
    filter = BayesFilter(initial_prob=0.5, process_var=0.01, measurement_var=0.05)
    
    # Einige Beispieldaten
    measurements = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 0.9, 0.7, 0.4, 0.2, 0.1]
    
    # Anwendung des Bayes-Filters auf die Beispieldaten
    filtered_values = [filter.process(measurement) for measurement in measurements]
    
    # Ergebnisse anzeigen
    for measurement, filtered_value in zip(measurements, filtered_values):
        print(f"Messung: {measurement:.2f} -> Gefilterter Wert: {filtered_value:.2f}")

if __name__ == "__main__":
    # Führe die Demo aus
    demo_bayes_filter()