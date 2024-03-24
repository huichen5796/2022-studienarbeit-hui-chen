import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SeiteMainExactorComponent } from './seite-main-exactor.component';

describe('SeiteMainExactorComponent', () => {
  let component: SeiteMainExactorComponent;
  let fixture: ComponentFixture<SeiteMainExactorComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [SeiteMainExactorComponent]
    });
    fixture = TestBed.createComponent(SeiteMainExactorComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
